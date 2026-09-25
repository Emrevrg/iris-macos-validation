import json, platform, sys
from datetime import datetime, timezone
import torch, transformers
from iris import verify_bit_exact

report = {
    "schema": "iris.macos-validation.v1",
    "measured": False,
    "bit_exact": False,
    "platform": {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    },
    "model": "Qwen/Qwen2.5-0.5B",
    "verification": None,
}
try:
    result = verify_bit_exact(
        "Qwen/Qwen2.5-0.5B",
        prompt="The capital of France is",
        device="cpu",
        low_mem=True,
        tensor_shards=1,
        gen_tokens=2,
        source="hub",
        reference="full",
    )
    report["verification"] = result
    report["measured"] = True
    report["bit_exact"] = (
        result.get("bit_exact_fraction") == 1.0
        and result.get("max_logit_abs_diff") == 0.0
        and result.get("next_token_identical") is True
    )
except Exception as exc:
    report["error"] = {"type": type(exc).__name__, "message": str(exc)}
report["generated_at"] = datetime.now(timezone.utc).isoformat()
with open("iris_macos_validation.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
if not report["measured"] or not report["bit_exact"]:
    raise SystemExit(1)