"""IRIS dağıtık HTTP serve — her node modelin bir katman-aralığını tutar, hidden'i HTTP ile
alır/verir; koordinatör node'ları zincirler. Aynı ağırlık + aynı hesap → çıktı bit-birebir.

Node (kendi katmanları + sınır parçaları):
    iris serve-node --model Qwen/Qwen2.5-7B --layers 0:14 --port 8001 --first
    iris serve-node --model Qwen/Qwen2.5-7B --layers 14:28 --port 8002 --last
Koordinatör:
    iris serve --distributed --model Qwen/Qwen2.5-7B --nodes http://localhost:8001,http://localhost:8002
"""
from __future__ import annotations
import os, json, base64, io, glob, math
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _tensor_b64(t):
    import torch
    buf = io.BytesIO(); torch.save(t, buf); return base64.b64encode(buf.getvalue()).decode()

def _b64_tensor(s):
    import torch
    return torch.load(io.BytesIO(base64.b64decode(s)), map_location="cpu")


class NodeShard:
    """Modelin [a:b] katmanlarini (+ ilkse embed/rope, sonsa norm/head) tutar, kismi forward yapar."""
    def __init__(self, model_id, a, b, first, last, device=None):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from accelerate import init_empty_weights
        self.torch = torch; self.a, self.b, self.first, self.last = a, b, first, last
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = AutoConfig.from_pretrained(model_id); self.cfg._attn_implementation = "eager"
        path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
        self.k2f = {}
        for f in sorted(glob.glob(path + "/*.safetensors")):
            with safe_open(f, "pt") as h:
                for k in h.keys(): self.k2f[k] = f
        self.safe_open = safe_open
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.cfg)
        self.model.eval(); mm = self.model.model
        if first:
            self._mat(mm.embed_tokens, "model.embed_tokens")
        if last:
            self._mat(mm.norm, "model.norm")
            if getattr(self.cfg, "tie_word_embeddings", False):
                self._mat(mm.embed_tokens, "model.embed_tokens")
                self.model.lm_head.weight = mm.embed_tokens.weight
            else:
                self._mat(self.model.lm_head, "lm_head")
        for i in range(a, b):
            self._mat(mm.layers[i], f"model.layers.{i}")
        # rope: kendi sinifindan yeniden ornekle (surume-dogru, bit-birebir)
        rot = getattr(mm, "rotary_emb", None)
        if rot is not None:
            try:
                mm.rotary_emb = type(rot)(config=self.cfg).to(self.device)
            except Exception:
                pass
        self.mm = mm

    def _get(self, k):
        with self.safe_open(self.k2f[k], "pt") as h:
            return h.get_tensor(k).to(self.device)

    def _mat(self, module, prefix):
        sd = {}
        for k in self.k2f:
            if k == prefix or k.startswith(prefix + "."):
                sd[k[len(prefix):].lstrip(".")] = self._get(k)
        if sd: module.load_state_dict(sd, assign=True)

    def forward(self, ids=None, hidden=None):
        torch = self.torch; mm = self.mm
        with torch.no_grad():
            if self.first:
                ids = ids.to(self.device)
                h = mm.embed_tokens(ids)
                T = ids.shape[1]
            else:
                h = hidden.to(self.device)
                T = h.shape[1]
            pos = torch.arange(T, device=self.device).unsqueeze(0)
            pe = mm.rotary_emb(h, pos)
            # nedensel maske (eager): ust-ucgen -inf
            mask = torch.full((T, T), float("-inf"), device=self.device)
            mask = torch.triu(mask, diagonal=1)[None, None]
            for i in range(self.a, self.b):
                out = mm.layers[i](h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
                h = out[0] if isinstance(out, (tuple, list)) else out
            if self.last:
                h = mm.norm(h)
                logits = self.model.lm_head(h)
                return {"logits_last": logits[0, -1].float().cpu().tolist()}
            return {"hidden": _tensor_b64(h.cpu())}


def serve_node(model_id, a, b, port, first, last):
    """Bir node'u HTTP sunucusu olarak baslat: POST /forward."""
    import http.server, socketserver
    from transformers import AutoTokenizer
    shard = NodeShard(model_id, a, b, first, last)
    tok = AutoTokenizer.from_pretrained(model_id) if first else None
    import torch

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if first:
                ids = torch.tensor(req["input_ids"])
                res = shard.forward(ids=ids)
            else:
                res = shard.forward(hidden=_b64_tensor(req["hidden"]))
            body = json.dumps(res).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
    print(f"[iris-node] {model_id} layers[{a}:{b}] first={first} last={last} :{port}", flush=True)
    with socketserver.TCPServer(("", port), H) as s:
        s.serve_forever()


def distributed_generate(model_id, node_urls, prompt, max_new_tokens=20):
    """Koordinatör: input_ids -> node0 -> ... -> son node -> logits -> örnekle -> tekrar."""
    import urllib.request
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()

    def post(url, payload):
        req = urllib.request.Request(url + "/forward", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())

    out_ids = list(ids)
    for _ in range(max_new_tokens):
        payload = {"input_ids": [out_ids]}
        for i, url in enumerate(node_urls):
            res = post(url, payload)
            payload = {"hidden": res["hidden"]} if "hidden" in res else res
        logits = res["logits_last"]
        nxt = int(max(range(len(logits)), key=lambda j: logits[j]))
        out_ids.append(nxt)
        if nxt == tok.eos_token_id: break
    return tok.decode(out_ids[len(ids):], skip_special_tokens=True)
