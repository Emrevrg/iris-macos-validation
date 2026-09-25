"""IRIS engine — kayipsiz katman/tensor akisi ile HF causal-LM'lerini sinirli bellekte
calistirir. Model bitleri atilmaz: agirliklar (yerel safetensors VEYA dogrudan Hub'dan HTTP
Range ile) katman katman / dilim dilim okunur, kullanilinca birakilir.

v0.2:
  * KV-cache'li uretim (token basina O(1) hesap, prompt yeniden hesaplanmaz)
  * prefetch: sonraki katman/dilim arka planda okunur (I/O ile hesap ortusur)
  * tensor_shards=N: her Linear N satir-dilimine bolunur (kolon-paralel), her dilim ayri
    yuklenir/birakilir -> yerlesik agirlik = katman/N. Cikti dilimlerin birlestirmesi.
  * source="hub": diske hic yazmadan Hub'dan dilim dilim akis (disk < model boyutu olsa bile)
  * verify: coklu istem, TUM pozisyonlar + KV-cache'li adim adim uretim logitleri

Bit-kesinlik yalniz ``verify_bit_exact`` ayni model/istem/dtype/cihazda 0.0 verdiginde
iddia edilir.

    from iris.engine import IrisModel
    m = IrisModel("Qwen/Qwen2.5-7B")
    print(m.generate("Merhaba, ", max_new_tokens=40))
"""
from __future__ import annotations
import os, glob, math, json, time, threading
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_DT = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}
_DSZ = {"F16": 2, "BF16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "U8": 1}


def _rss_gb():
    """Anlik resident bellek (VmRSS) — yuksek-su-isareti degil."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return 0.0


# ---------------------------------------------------------------- agirlik kaynaklari
class LocalSource:
    """Yerel safetensors (snapshot_download). Dilim okuma mmap ile yalniz gereken sayfalar."""

    def __init__(self, model_id):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        self._safe_open = safe_open
        path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
        self.meta = {}
        for f in sorted(glob.glob(path + "/*.safetensors")):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    sl = h.get_slice(k)
                    self.meta[k] = (f, sl.get_dtype(), list(sl.get_shape()))
        if not self.meta:
            raise RuntimeError("safetensors bulunamadi: " + model_id)
        self.bytes_read = 0

    def keys(self):
        return self.meta.keys()

    def shape(self, key):
        return self.meta[key][2]

    def get(self, key):
        with self._safe_open(self.meta[key][0], "pt") as h:
            t = h.get_tensor(key)
        self.bytes_read += t.numel() * t.element_size()
        return t

    def get_rows(self, key, r0, r1):
        with self._safe_open(self.meta[key][0], "pt") as h:
            t = h.get_slice(key)[r0:r1]
        self.bytes_read += t.numel() * t.element_size()
        return t

    def get_row_list(self, key, idx):
        import torch
        with self._safe_open(self.meta[key][0], "pt") as h:
            sl = h.get_slice(key)
            t = torch.stack([sl[int(i)] for i in idx])
        self.bytes_read += t.numel() * t.element_size()
        return t


class RamSource(LocalSource):
    """Veri-merkezi modu: tum agirliklar bir kez SABITLENMIS (pinned) host RAM'e alinir,
    GPU'da yalniz o an hesaplanan katman/dilim durur. Host->GPU kopyasi pinned bellekten
    (~PCIe hizi) ve asenkron; disk token basina hic okunmaz."""

    def __init__(self, model_id):
        super().__init__(model_id)
        import torch
        pin = torch.cuda.is_available()
        self.cache = {}
        for k in self.meta:
            t = super().get(k)
            self.cache[k] = t.pin_memory() if pin else t
        self.bytes_read = 0

    def get(self, key):
        t = self.cache[key]
        self.bytes_read += t.numel() * t.element_size()
        return t

    def get_rows(self, key, r0, r1):
        t = self.cache[key][r0:r1]
        self.bytes_read += t.numel() * t.element_size()
        return t

    def get_row_list(self, key, idx):
        import torch
        t = self.cache[key][torch.as_tensor([int(i) for i in idx])]
        self.bytes_read += t.numel() * t.element_size()
        return t


class HubSource:
    """Diske YAZMADAN Hub'dan akis: safetensors basligi + HTTP Range ile tam gereken baytlar.
    Disk kapasitesi modelden kucuk olsa bile (orn. 145 GB model, 20 GB disk) calisir."""

    def __init__(self, model_id, revision="main", token=None, chunk_mb=64):
        import requests
        self.s = requests.Session()
        self.hdr = {}
        token = token or os.environ.get("HF_TOKEN")
        if token:
            self.hdr["Authorization"] = "Bearer " + token
        self.base = f"https://huggingface.co/{model_id}/resolve/{revision}/"
        self.chunk = chunk_mb * 1024 * 1024
        r = self.s.get(self.base + "model.safetensors.index.json", headers=self.hdr)
        files = sorted(set(r.json()["weight_map"].values())) if r.status_code == 200 \
            else ["model.safetensors"]
        self.meta = {}
        for f in files:
            n = int.from_bytes(self._range(f, 0, 8), "little")
            head = json.loads(self._range(f, 8, 8 + n))
            for k, v in head.items():
                if k == "__metadata__":
                    continue
                b, e = v["data_offsets"]
                self.meta[k] = (f, v["dtype"], v["shape"], 8 + n + b, 8 + n + e)
        self.bytes_read = 0
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=8)

    def _url(self, f, refresh=False):
        """resolve/ -> CDN yonlendirmesini dosya basina bir kez coz (Hub'a binlerce istek
        gitmesin, rate-limit yok); imzali URL ~20 dk'da bir veya hata olunca yenilenir."""
        c = getattr(self, "_urls", None)
        if c is None:
            c = self._urls = {}
        u = c.get(f)
        if u and not refresh and time.time() - u[1] < 1200:
            return u[0]
        r = self.s.get(self.base + f, headers=dict(self.hdr, Range="bytes=0-0"),
                       allow_redirects=False, timeout=60)
        loc = r.headers.get("Location") if 300 <= r.status_code < 400 else None
        if loc and loc.startswith("/"):
            loc = "https://huggingface.co" + loc
        c[f] = (loc or self.base + f, time.time())
        return c[f][0]

    def _range(self, f, a, b, tries=6):
        h = {"Range": f"bytes={a}-{b - 1}"}
        for t in range(tries):
            try:
                url = self._url(f, refresh=t > 0)
                hh = h if url.startswith("https://huggingface.co/") is False else dict(self.hdr, **h)
                r = self.s.get(url, headers=hh, timeout=120)
                if r.status_code in (200, 206) and len(r.content) == b - a:
                    return r.content
                if r.status_code == 200 and a == 0:
                    return r.content[:b]
            except Exception:
                pass
            time.sleep(1.5 * (t + 1))
        raise IOError(f"Range okunamadi {f} {a}-{b}")

    def _bytes(self, f, a, b):
        if b - a <= self.chunk:
            return self._range(f, a, b)
        cuts = list(range(a, b, self.chunk)) + [b]
        parts = list(self._pool.map(lambda ab: self._range(f, *ab), zip(cuts[:-1], cuts[1:])))
        return b"".join(parts)

    def _tensor(self, buf, dt, shape):
        import torch
        t = torch.frombuffer(bytearray(buf), dtype=getattr(torch, _DT[dt]))
        self.bytes_read += len(buf)
        return t.reshape(shape)

    def keys(self):
        return self.meta.keys()

    def shape(self, key):
        return list(self.meta[key][2])

    def get(self, key):
        f, dt, shape, a, b = self.meta[key]
        return self._tensor(self._bytes(f, a, b), dt, shape)

    def get_rows(self, key, r0, r1):
        f, dt, shape, a, b = self.meta[key]
        row = _DSZ[dt] * (math.prod(shape[1:]) if len(shape) > 1 else 1)
        return self._tensor(self._bytes(f, a + r0 * row, a + r1 * row), dt, [r1 - r0] + shape[1:])

    def get_row_list(self, key, idx):
        import torch
        return torch.cat([self.get_rows(key, int(i), int(i) + 1) for i in idx])


# ---------------------------------------------------------------- akis zamanlayici
class _Streamer:
    """Bir forward'in agirlik isteklerini SIRASIYLA bilir; `window` is ileriyi arka planda
    getirir. Yanlis sira sadece prefetch verimini dusurur, dogrulugu asla (is anahtarla alinir)."""

    def __init__(self, loader, plan, window=2, workers=2):
        from concurrent.futures import ThreadPoolExecutor
        self.loader, self.plan, self.window = loader, plan, window
        self.pos = {j: i for i, j in enumerate(plan)}
        self.pending = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers)) if window > 0 else None

    def take(self, job):
        with self.lock:
            fut = self.pending.pop(job, None)
        out = fut.result() if fut is not None else self.loader(job)
        if self.pool is not None and job in self.pos:
            p, n = self.pos[job], len(self.plan)
            want = [self.plan[(p + k) % n] for k in range(1, self.window + 1)]
            with self.lock:
                for j in list(self.pending):
                    if j not in want:
                        self.pending.pop(j).cancel()
                for j in want:
                    if j not in self.pending and j != job:
                        self.pending[j] = self.pool.submit(self.loader, j)
        return out

    def reset(self):
        with self.lock:
            for f in self.pending.values():
                f.cancel()
            self.pending.clear()


def _make_sharded_linear():
    import torch
    import torch.nn.functional as F

    class ShardedLinear(torch.nn.Module):
        """nn.Linear yerine: agirlik N satir-dilimi halinde akistan alinir. N=1 -> birebir
        nn.Linear.forward (F.linear, tam matris)."""

        def __init__(self, host, layer, name, out_features, in_features, has_bias, n):
            super().__init__()
            self.host, self.layer, self.name = host, layer, name
            self.out_features, self.in_features, self.has_bias, self.n = \
                out_features, in_features, has_bias, n

            self.lora_A = self.lora_B = None
            self.lora_scale = 0.0

        def add_lora(self, rank, alpha, device, dtype, seed):
            """Egitilebilir dusuk-rank adaptor (yerlesik, kucuk). Taban agirlik akista kalir,
            hic degismez. B sifirla baslar -> eklenince cikti tabanla BIREBIR ayni."""
            g = torch.Generator().manual_seed(seed)
            A = torch.randn(rank, self.in_features, generator=g) / math.sqrt(self.in_features)
            self.lora_A = torch.nn.Parameter(A.to(device=device, dtype=dtype))
            self.lora_B = torch.nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
            self.lora_scale = alpha / rank

        def forward(self, x):
            outs = []
            for s in range(self.n):
                W, b = self.host._stream.take(("lin", self.layer, self.name, s))
                outs.append(F.linear(x, W, b))
                del W, b
            y = outs[0] if self.n == 1 else torch.cat(outs, -1)
            if self.lora_A is not None:
                d = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B) * self.lora_scale
                y = y + d.to(y.dtype)
            return y

        def extra_repr(self):
            return f"{self.in_features}->{self.out_features}, shards={self.n}"

    return ShardedLinear


# ---------------------------------------------------------------- model
class IrisModel:
    """Bir HF causal-LM'i kayipsiz, katman/tensor-akisli calistirir."""

    def __init__(self, model_id: str, device: str | None = None, dtype=None, low_mem: bool = False,
                 tensor_shards: int = 1, source: str = "local", prefetch: int | None = None):
        import torch
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
        from accelerate import init_empty_weights

        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.cfg = AutoConfig.from_pretrained(model_id)
        self.cfg._attn_implementation = "eager"   # deterministik + surum-kararli
        self.attn_impl = "eager"
        self.low_mem, self.tensor_shards = low_mem, max(1, int(tensor_shards))
        self.src = {"hub": HubSource, "ram": RamSource}.get(source, LocalSource)(model_id)
        self.source = source
        if prefetch is None:
            prefetch = 16 if source == "hub" else 2

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.cfg)
        self.model.eval()
        mm = self.model.model
        L = len(mm.layers)

        # her katmandaki Linear'lari ShardedLinear ile degistir; kalan (norm vb.) katman isi
        SL = _make_sharded_linear()
        self._lin_names = []
        for i, layer in enumerate(mm.layers):
            names = [n for n, mod in layer.named_modules() if isinstance(mod, torch.nn.Linear)]
            if i == 0:
                self._lin_names = names
            for n in names:
                mod = layer.get_submodule(n)
                parent = layer.get_submodule(n.rsplit(".", 1)[0]) if "." in n else layer
                setattr(parent, n.rsplit(".", 1)[-1],
                        SL(self, i, n, mod.out_features, mod.in_features, mod.bias is not None,
                           self.tensor_shards))
        self._layer_other = []
        for i in range(L):
            pre = f"model.layers.{i}."
            lin = tuple(pre + n + "." for n in self._lin_names)
            self._layer_other.append([k for k in self.src.keys()
                                      if k.startswith(pre) and not k.startswith(lin)])

        plan = []
        for i in range(L):
            plan.append(("layer", i))
            for n in self._lin_names:
                plan += [("lin", i, n, s) for s in range(self.tensor_shards)]
        self._stream = _Streamer(self._load_job, plan, window=prefetch,
                                 workers=8 if source == "hub" else 2)

        self._materialize(mm.norm, "model.norm")
        tie = getattr(self.cfg, "tie_word_embeddings", False)
        embed_key = "model.embed_tokens.weight"
        head_key = embed_key if tie else "lm_head.weight"
        if low_mem:
            self._install_low_mem(mm, embed_key, head_key)
        else:
            self._materialize(mm.embed_tokens, "model.embed_tokens")
            if tie:
                self.model.lm_head.weight = mm.embed_tokens.weight
            else:
                self._materialize(self.model.lm_head, "lm_head")
        self._init_rope(mm)

        for i, layer in enumerate(mm.layers):
            layer.register_forward_pre_hook(lambda m, a, idx=i: self._enter_layer(m, idx))
            layer.register_forward_hook(lambda m, a, o: (self._free(m), None)[1])
        self.peak_gb = 0.0
        self.last_stats = {}

    # --- dahili ---
    def _load_job(self, job):
        torch = self.torch
        if job[0] == "layer":
            pre = f"model.layers.{job[1]}."
            return {k[len(pre):]: self.src.get(k).to(self.device) for k in self._layer_other[job[1]]}
        _, i, n, s = job
        key = f"model.layers.{i}.{n}"
        out = self.src.shape(key + ".weight")[0]
        step = math.ceil(out / self.tensor_shards)
        r0, r1 = s * step, min((s + 1) * step, out)
        nb = self.device != "cpu"
        W = self.src.get_rows(key + ".weight", r0, r1).to(self.device, non_blocking=nb)
        b = self.src.get_rows(key + ".bias", r0, r1).to(self.device, non_blocking=nb) \
            if (key + ".bias") in self.src.meta else None
        return W, b

    def _enter_layer(self, module, idx):
        sd = self._stream.take(("layer", idx))
        if sd:
            module.load_state_dict(sd, assign=True, strict=False)
        return None

    def _install_low_mem(self, mm, embed_key, head_key, blocks=8):
        """embed/lm_head'i RESIDENT tutmadan akistan oku (forward override). Head TUM
        pozisyonlar icin logit dondurur (kolon-bloklari = lm_head'in tensor-sharding'i)."""
        import torch
        import torch.nn.functional as F
        host = self
        D, V = self.cfg.hidden_size, self.cfg.vocab_size
        V = self.src.shape(head_key)[0]

        def tiny_embed_forward(x):
            flat = x.flatten().tolist()
            uniq = sorted(set(flat))
            rows = host.src.get_row_list(embed_key, uniq).to(host.device)
            rmap = {t: i for i, t in enumerate(uniq)}
            out = torch.stack([rows[rmap[t]] for t in flat])
            return out.view(*x.shape, D).to(rows.dtype)

        def block_head_forward(hidden):
            step = math.ceil(V / blocks)
            outs = []
            for b in range(0, V, step):
                Wb = host.src.get_rows(head_key, b, min(b + step, V)).to(host.device).to(hidden.dtype)
                outs.append(F.linear(hidden, Wb))
                del Wb
            return torch.cat(outs, -1)

        mm.embed_tokens.forward = tiny_embed_forward
        self.model.lm_head.forward = block_head_forward

    def _materialize(self, module, prefix):
        sd = {}
        for k in self.src.keys():
            if k == prefix or k.startswith(prefix + "."):
                sd[k[len(prefix):].lstrip(".")] = self.src.get(k).to(self.device)
        if sd:
            module.load_state_dict(sd, assign=True, strict=False)
        return len(sd)

    def _init_rope(self, mm):
        """rotary_emb'i KENDI sinifindan gercek cihazda yeniden ornekle (surume-dogru inv_freq)."""
        torch = self.torch
        rot = getattr(mm, "rotary_emb", None)
        if rot is None:
            return
        try:
            try:
                new_rot = type(rot)(config=self.cfg)
            except TypeError:
                hd = self.cfg.hidden_size // self.cfg.num_attention_heads
                new_rot = type(rot)(hd, getattr(self.cfg, "max_position_embeddings", 4096),
                                    getattr(self.cfg, "rope_theta", 10000.0))
            mm.rotary_emb = new_rot.to(self.device)
            return
        except Exception:
            pass
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        rs = getattr(self.cfg, "rope_scaling", None) or {}
        fn = ROPE_INIT_FUNCTIONS.get(rs.get("rope_type", rs.get("type", "default"))) \
            or ROPE_INIT_FUNCTIONS["default"]
        inv, scale = fn(self.cfg, self.device)
        rot.inv_freq = inv.to(self.device)
        rot.attention_scaling = scale
        rot.original_inv_freq = rot.inv_freq

    def _free(self, module):
        """Akistan gelen (taban) parametreleri birak; egitilebilir LoRA parametrelerine dokunma."""
        torch = self.torch
        for mod in module.modules():
            for pn, p in list(mod._parameters.items()):
                if p is None or pn.startswith("lora_") or p.device.type == "meta":
                    continue
                mod._parameters[pn] = torch.nn.Parameter(torch.empty_like(p, device="meta"),
                                                         requires_grad=False)

    # --- ince ayar (taban akista, adaptor yerlesik) ---
    def add_lora(self, rank=8, alpha=16, targets=("q_proj", "v_proj"), seed=0, dtype=None):
        torch = self.torch
        dtype = dtype or torch.float32
        params = []
        for i, layer in enumerate(self.model.model.layers):
            for n in self._lin_names:
                if n.rsplit(".", 1)[-1] in targets:
                    mod = layer.get_submodule(n)
                    import zlib
                    mod.add_lora(rank, alpha, self.device, dtype,
                                 seed * 100003 + i * 131 + zlib.crc32(n.encode()) % 9973)
                    params += [mod.lora_A, mod.lora_B]
        return params

    def lora_state(self):
        return {k: v.detach().clone() for k, v in self.model.named_parameters() if ".lora_" in k}

    def load_lora_state(self, sd):
        cur = dict(self.model.named_parameters())
        with self.torch.no_grad():
            for k, v in sd.items():
                cur[k].copy_(v)

    def loss(self, input_ids):
        """Nedensel LM kaybi, gradyanli (yalniz adaptorlere akar)."""
        torch = self.torch
        ids = input_ids.to(self.device)
        logits = self.model(ids, use_cache=False).logits.float()
        return torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])

    def _chat_default(self):
        n = self.model_id.lower()
        return bool(getattr(self.tok, "chat_template", None)) and \
            any(t in n for t in ("instruct", "chat", "-it", "assistant"))

    def encode(self, prompt, chat=None):
        chat = self._chat_default() if chat is None else chat
        if chat and getattr(self.tok, "chat_template", None):
            enc = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True)
            return enc["input_ids"]
        return self.tok(prompt, return_tensors="pt").input_ids

    # --- genel API ---
    def forward_logits(self, input_ids, past_key_values=None):
        torch = self.torch
        with torch.no_grad():
            out = self.model(input_ids.to(self.device), past_key_values=past_key_values,
                             use_cache=past_key_values is not None)
        self.peak_gb = max(self.peak_gb, round(_rss_gb(), 3))
        return out.logits

    def generate(self, prompt, max_new_tokens=40, chat=None, return_logits=False, ids=None):
        """Aciksoz (greedy) uretim, KV-cache'li: prompt bir kez, sonra token basina tek adim."""
        torch = self.torch
        from transformers import DynamicCache
        ids = self.encode(prompt, chat) if ids is None else ids
        cache = DynamicCache()
        eos = self.tok.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        gen, steps, cur = [], [], ids
        b0, t0 = self.src.bytes_read, time.time()
        t_first = None
        for _ in range(max_new_tokens):
            logits = self.forward_logits(cur, past_key_values=cache)[:, -1].float().cpu()
            if t_first is None:
                t_first = time.time() - t0
            if return_logits:
                steps.append(logits[0])
            nxt = int(logits.argmax(-1))
            gen.append(nxt)
            if nxt in eos:
                break
            cur = torch.tensor([[nxt]])
        dt = time.time() - t0
        self.last_stats = {
            "new_tokens": len(gen), "seconds": round(dt, 3),
            "first_token_s": round(t_first or 0, 3),
            "decode_tok_per_s": round((len(gen) - 1) / max(dt - (t_first or 0), 1e-9), 4)
            if len(gen) > 1 else None,
            "weight_gb_read": round((self.src.bytes_read - b0) / 1e9, 3),
            "peak_rss_gb": self.peak_gb,
        }
        text = self.tok.decode(gen, skip_special_tokens=True)
        return (text, gen, steps) if return_logits else text


def shard_plan(model_id, nodes=8):
    """N node'a bolme plani (agirlik yuklemeden, safetensors basligindan). Calisan tensor
    sharding icin: IrisModel(..., tensor_shards=N)."""
    import re
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    L = cfg.num_hidden_layers
    src = LocalSource(model_id)
    layer_bytes, other = [0] * L, 0
    for k, (f, dt, shp) in src.meta.items():
        nb = math.prod(shp) * _DSZ.get(dt, 2)
        m = re.search(r"layers\.(\d+)\.", k)
        if m and int(m.group(1)) < L:
            layer_bytes[int(m.group(1))] += nb
        else:
            other += nb
    total = sum(layer_bytes) + other
    per = math.ceil(L / nodes)
    groups = [list(range(i, min(i + per, L))) for i in range(0, L, per)]
    node_bytes = [sum(layer_bytes[i] for i in g) for g in groups]
    node_bytes[0] += other
    mx = max(node_bytes)
    tp = math.ceil(total / max(nodes, 1))
    return {
        "model": model_id, "layers": L, "nodes": min(len(groups), nodes),
        "model_gb": round(total / 1e9, 2),
        "per_node_gb": round(mx / 1e9, 3), "reduction_x": round(total / max(mx, 1), 1),
        "groups": [[g[0], g[-1]] for g in groups],
        "tensor_shard_per_node_gb": round(tp / 1e9, 4),
        "tensor_shard_reduction_x": round(total / max(tp, 1), 1),
        "bit_exact_status": "plan; olcum icin: iris verify --tensor-shards N",
    }


def _ref_greedy(model, ids, n, device, eos):
    import torch
    from transformers import DynamicCache
    cache, cur, gen, steps = DynamicCache(), ids, [], []
    with torch.no_grad():
        for _ in range(n):
            lg = model(cur.to(device), past_key_values=cache, use_cache=True).logits[:, -1].float().cpu()
            steps.append(lg[0])
            nxt = int(lg.argmax(-1))
            gen.append(nxt)
            if nxt in eos:
                break
            cur = torch.tensor([[nxt]])
    return gen, steps


DEFAULT_PROMPTS = [
    "The theory of relativity",
    "Merhaba, bugün hava çok güzel ve ben",
    "def fibonacci(n):\n    ",
    "1, 1, 2, 3, 5, 8, 13,",
]


def verify_bit_exact(model_id, prompt=None, prompts=None, device=None, low_mem=False,
                     tensor_shards=1, gen_tokens=8, source="local", reference="full"):
    """IRIS ciktisini referansla karsilastir: her istemde TUM pozisyon logitleri + KV-cache'li
    adim adim uretim (token + logit). reference="full": tam-yuklu HF modeli;
    reference="iris-layer": ayni motorun tensor_shards=1 yolu (tam model sigmiyorsa)."""
    import torch
    prompts = prompts or ([prompt] if prompt else DEFAULT_PROMPTS)
    m = IrisModel(model_id, device=device, low_mem=low_mem, tensor_shards=tensor_shards, source=source)
    eos = m.tok.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    ours = []
    for p in prompts:
        ids = m.tok(p, return_tensors="pt").input_ids
        full = m.forward_logits(ids)[0].float().cpu()
        _, gen, steps = m.generate(None, max_new_tokens=gen_tokens, ids=ids, return_logits=True)
        ours.append((ids, full, gen, steps))
    peak = m.peak_gb
    stats = m.last_stats
    del m
    if reference == "full":
        from transformers import AutoModelForCausalLM
        ref = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="eager").to(
            device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
        dev = next(ref.parameters()).device
        fwd = lambda ids: ref(ids.to(dev)).logits[0].float().cpu()
        gen_ref = lambda ids: _ref_greedy(ref, ids, gen_tokens, dev, eos)
    else:
        r = IrisModel(model_id, device=device, low_mem=low_mem, tensor_shards=1, source=source)
        fwd = lambda ids: r.forward_logits(ids)[0].float().cpu()
        gen_ref = lambda ids: r.generate(None, max_new_tokens=gen_tokens, ids=ids, return_logits=True)[1:]
    per, mx, exact, n = [], 0.0, 0.0, 0
    tok_all = True
    for p, (ids, full, gen, steps) in zip(prompts, ours):
        with torch.no_grad():
            rf = fwd(ids)
            rg, rs = gen_ref(ids)
        d_full = (full - rf).abs().max().item()
        d_gen = max(((a - b).abs().max().item() for a, b in zip(steps, rs)), default=0.0)
        same = gen == rg
        tok_all &= same
        mx = max(mx, d_full, d_gen)
        exact += (full == rf).float().sum().item() + sum((a == b).float().sum().item()
                                                         for a, b in zip(steps, rs))
        n += full.numel() + sum(a.numel() for a in steps)
        per.append({"prompt": p[:40], "positions": ids.shape[1], "full_seq_max_diff": d_full,
                    "gen_steps": len(gen), "gen_logit_max_diff": d_gen, "gen_tokens_identical": same})
    return {
        "model": model_id, "mode": {"low_mem": low_mem, "tensor_shards": tensor_shards,
                                    "source": source, "reference": reference, "kv_cache": True},
        "peak_gb": peak, "prompts": per,
        "max_logit_abs_diff": mx, "bit_exact_fraction": exact / max(n, 1),
        "logits_compared": n, "next_token_identical": tok_all, "gen_tokens_identical": tok_all,
        "last_gen_stats": stats,
    }
