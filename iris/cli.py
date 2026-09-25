"""IRIS komut satiri: herhangi bir modeli kayipsiz, katman/tensor-akisli calistir.

    iris run Qwen/Qwen2.5-7B --prompt "Merhaba" --max-new 40
    iris verify Qwen/Qwen2.5-7B                     # tam-yukle ile bit-birebir mi?
    iris verify Qwen/Qwen2.5-7B --tensor-shards 8   # dilim-akisi da bit-birebir mi?
    iris run Qwen/Qwen2.5-72B --source hub --low-mem --tensor-shards 8   # diske yazmadan
    iris bench Qwen/Qwen2.5-7B                      # ilk-token, decode tok/s, GB/token
"""
from __future__ import annotations
import argparse, json, sys


def _common(p):
    p.add_argument("model")
    p.add_argument("--device", default=None)
    p.add_argument("--low-mem", action="store_true", help="embed/lm_head'i akistan oku")
    p.add_argument("--tensor-shards", type=int, default=1,
                   help="her Linear'i N dilimde akit (yerlesik agirlik = katman/N)")
    p.add_argument("--source", choices=["local", "hub", "ram"], default="local",
                   help="hub: diske yazmadan HTTP Range ile Hub'dan akit; ram: agirliklar "
                        "pinned host RAM'de, GPU'da yalniz aktif katman (veri-merkezi modu)")


def _chat_flag(a):
    return True if a.chat else (False if a.no_chat else None)


def main(argv=None):
    p = argparse.ArgumentParser(prog="iris", description="IRIS — kayipsiz dagitik model motoru")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="modeli akisla calistir ve metin uret (KV-cache'li)")
    _common(r)
    r.add_argument("--prompt", default="Merhaba, kısaca kendini tanıt.")
    r.add_argument("--max-new", type=int, default=40)
    r.add_argument("--chat", action="store_true", help="chat template'i zorla")
    r.add_argument("--no-chat", action="store_true", help="chat template kullanma "
                   "(varsayilan: yalniz instruct/chat modellerinde kullanilir)")

    v = sub.add_parser("verify", help="cikti referansla bit-birebir mi olc")
    _common(v)
    v.add_argument("--prompt", default=None, help="varsayilan: 4 farkli istem")
    v.add_argument("--gen-tokens", type=int, default=8)
    v.add_argument("--reference", choices=["full", "iris-layer"], default="full",
                   help="iris-layer: tam model belleğe sigmiyorsa motorun tensor_shards=1 yolu")

    b = sub.add_parser("bench", help="hiz: ilk-token, decode tok/s, token basina okunan GB")
    _common(b)
    b.add_argument("--prompt", default="The theory of relativity")
    b.add_argument("--max-new", type=int, default=16)

    s = sub.add_parser("serve", help="N node shard plani / dagitik koordinator")
    s.add_argument("model")
    s.add_argument("--nodes", type=int, default=8)
    s.add_argument("--run", action="store_true", help="tek makinede calistir + uret")
    s.add_argument("--prompt", default="Merhaba, kısaca kendini tanıt.")
    s.add_argument("--distributed", action="store_true", help="gercek cok-node HTTP koordinatoru")
    s.add_argument("--node-urls", default="", help="virgulle ayrik node URL'leri")
    s.add_argument("--max-new", type=int, default=20)

    sn = sub.add_parser("serve-node", help="bir dagitik node baslat (kendi katman araligi, HTTP)")
    sn.add_argument("model")
    sn.add_argument("--layers", required=True, help="a:b katman araligi, orn 0:14")
    sn.add_argument("--port", type=int, required=True)
    sn.add_argument("--first", action="store_true", help="ilk node (embed + tokenizer)")
    sn.add_argument("--last", action="store_true", help="son node (norm + lm_head)")

    a = p.parse_args(argv)

    if a.cmd in ("run", "bench"):
        from iris.engine import IrisModel
        m = IrisModel(a.model, device=a.device, low_mem=a.low_mem,
                      tensor_shards=a.tensor_shards, source=a.source)
        print(f"[iris] {a.model} · cihaz={m.device} · kaynak={a.source} · shards={a.tensor_shards}"
              + (" · low_mem" if a.low_mem else ""), flush=True)
        chat = _chat_flag(a) if a.cmd == "run" else False
        out = m.generate(a.prompt, max_new_tokens=a.max_new, chat=chat)
        print("\n" + out.strip() + "\n")
        st = m.last_stats
        print(json.dumps(st, ensure_ascii=False) if a.cmd == "bench" else
              f"[iris] tepe RSS {st['peak_rss_gb']} GB · ilk token {st['first_token_s']} s · "
              f"decode {st['decode_tok_per_s']} tok/s")
        return 0

    if a.cmd == "verify":
        from iris.engine import verify_bit_exact
        res = verify_bit_exact(a.model, prompt=a.prompt, device=a.device, low_mem=a.low_mem,
                               tensor_shards=a.tensor_shards, gen_tokens=a.gen_tokens,
                               source=a.source, reference=a.reference)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        ok = res["max_logit_abs_diff"] == 0.0 and res["bit_exact_fraction"] == 1.0 \
            and res["gen_tokens_identical"]
        print("[iris] ✓ BİT-BİREBİR (kayıpsız) · %d logit" % res["logits_compared"] if ok else
              f"[iris] ✗ token aynı={res['gen_tokens_identical']}, logit Δ={res['max_logit_abs_diff']}")
        return 0 if ok else 1

    if a.cmd == "serve-node":
        from iris.serve import serve_node
        x, y = a.layers.split(":")
        serve_node(a.model, int(x), int(y), a.port, a.first, a.last)
        return 0

    if a.cmd == "serve":
        if a.distributed:
            from iris.serve import distributed_generate
            urls = [u.strip() for u in a.node_urls.split(",") if u.strip()]
            out = distributed_generate(a.model, urls, a.prompt, max_new_tokens=a.max_new)
            print("\n" + out.strip() + "\n")
            print(f"[iris] {len(urls)} node üzerinden dağıtık üretim")
            return 0
        from iris.engine import shard_plan
        plan = shard_plan(a.model, nodes=a.nodes)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print(f"[iris] {plan['model']} ({plan['model_gb']} GB) → katman-sharding {plan['nodes']} node · "
              f"her node ~{plan['per_node_gb']} GB ({plan['reduction_x']}×)")
        print(f"[iris] tensör-sharding {a.nodes} dilim · dilim başı ~{plan['tensor_shard_per_node_gb']} GB "
              f"— çalıştır/doğrula: iris verify {a.model} --tensor-shards {a.nodes}")
        if a.run:
            from iris.engine import IrisModel
            m = IrisModel(a.model, tensor_shards=a.nodes)
            print("\n" + m.generate(a.prompt, max_new_tokens=32).strip())
            print(f"[iris] {a.nodes} dilim · tepe RSS {m.peak_gb} GB")
        return 0


if __name__ == "__main__":
    sys.exit(main())
