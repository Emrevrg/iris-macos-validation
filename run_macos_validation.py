import json, platform, sys, subprocess
from datetime import datetime, timezone
import torch, transformers
from iris import verify_bit_exact

report = {
    "schema": "iris.macos-validation.v2",
    "measured": False,
    "bit_exact": False,
    "platform": {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "mac_ver": platform.mac_ver()[0],
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    },
    "model": "Qwen/Qwen2.5-0.5B",
    "verifications": {},
}
try:
    report["platform"]["sysctl_cpu"] = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
except Exception:
    pass
modes = {
    "normal": dict(),
    "low_mem+shards4": dict(low_mem=True, tensor_shards=4),
}
try:
    for name, kw in modes.items():
        # varsayilan 4 istem, tum pozisyonlar + KV-cache'li 8 adim uretim; referans ayni makinede tam model
        r = verify_bit_exact("Qwen/Qwen2.5-0.5B", device="cpu", gen_tokens=8, reference="full", **kw)
        report["verifications"][name] = r
        print(name, r["max_logit_abs_diff"], r["bit_exact_fraction"], r["logits_compared"], flush=True)
    report["measured"] = platform.system() == "Darwin" and len(report["verifications"]) == len(modes)
    report["bit_exact"] = all(
        r["bit_exact_fraction"] == 1.0 and r["max_logit_abs_diff"] == 0.0 and r["gen_tokens_identical"] is True
        for r in report["verifications"].values())
    report["logits_compared"] = sum(r["logits_compared"] for r in report["verifications"].values())
except Exception as exc:
    import traceback
    report["error"] = {"type": type(exc).__name__, "message": str(exc), "trace": traceback.format_exc()[-2000:]}
report["generated_at"] = datetime.now(timezone.utc).isoformat()
with open("iris_macos_validation.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({k: report[k] for k in ("measured", "bit_exact", "platform")}, ensure_ascii=False, indent=2))
if not report["measured"] or not report["bit_exact"]:
    raise SystemExit(1)
