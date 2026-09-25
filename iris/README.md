---
license: apache-2.0
library_name: iris-engine
pipeline_tag: text-generation
tags:
  - inference
  - lossless
  - distributed
  - sharding
  - bit-exact
  - airllm
  - colibri
language:
  - tr
  - en
---

# IRIS — kayıpsız dağıtık model motoru

**Herhangi bir HF modelini, bellek bütçesi altında, bit-birebir (kayıpsız) çalıştır.**
Model bitleri hiç atılmaz; katman katman seçmeli yüklenir, kullanılınca bırakılır. Tepe bellek =
tek katman, model boyutu değil. Çıktı, tam-yükle referansla logit düzeyinde **birebir aynı** —
kendin `iris verify` ile doğrularsın.

> Bit **sıkıştırma** değil (yüksek-entropi ağırlıkların kayıpsız sınırı ~1,3× — fizik).
> Kayıpsız **dağıtım/akış**: yerleşik ayak-izi = model/N, N sınırsız.

## Kur

```bash
pip install git+https://huggingface.co/emrevrg/iris-engine
```

## Kullan

```bash
# çalıştır — tek katman kadar RAM/VRAM
iris run Qwen/Qwen2.5-7B --prompt "Görelilik nedir?" --max-new 40

# daha az resident (embed/lm_head'i diskten oku) — büyük-vocab modellerde 14B → <1 GB
iris run Qwen/Qwen2.5-14B --low-mem --prompt "..."
iris verify Qwen/Qwen2.5-14B --low-mem   # → bit-birebir 0.0

# doğrula — çıktı tam-modelle bit-birebir mi?
iris verify Qwen/Qwen2.5-7B
#   → max_logit_abs_diff: 0.0 · bit_exact_fraction: 1.0 · ✓ KAYIPSIZ

# dağıt — N node'a shard planı (tensör-shard ile N sınırsız)
iris serve Qwen/Qwen2.5-7B --nodes 8
```

Koddan:

```python
from iris.engine import IrisModel, verify_bit_exact, shard_plan

m = IrisModel("Qwen/Qwen2.5-7B")          # GPU varsa kullanır
print(m.generate("The theory of relativity", max_new_tokens=20, chat=False))
print("resident:", m.peak_gb, "GB")

verify_bit_exact("Qwen/Qwen2.5-7B")
# {'max_logit_abs_diff': 0.0, 'bit_exact_fraction': 1.0, 'next_token_identical': True}
```

## Gerçek çok-node HTTP serve

Modeli ayrı sunuculara böl, hidden state'i HTTP ile aktar — çıktı tam-modelle birebir:

```bash
# node A (ilk yarı + embed)
iris serve-node Qwen/Qwen2.5-7B --layers 0:14 --port 8001 --first
# node B (son yarı + head)  — başka bir makinede olabilir
iris serve-node Qwen/Qwen2.5-7B --layers 14:28 --port 8002 --last
# koordinatör
iris serve Qwen/Qwen2.5-7B --distributed \
  --node-urls http://localhost:8001,http://localhost:8002 --prompt "..."
```

## Ölçekleme yasası (ölçülen)

Yerleşik bellek ~tek katmanla sınırlı → **model büyüdükçe küçültme oranı lineer artar**:

| Model | Boyut | Resident RAM | Küçültme | Bit-birebir |
|---|---|---|---|---|
| Qwen2.5-7B | 15,2 GB | 1,78 GB | **8,5×** | ✓ 0.0 |
| Qwen2.5-14B | 29,5 GB | 2,63 GB | **11,2×** | ✓ |
| Qwen2.5-32B | 65,5 GB | 2,63 GB | **24,9×** | ✓ |
| ~263B (ekstrapolasyon) | ~526 GB | ~2,6 GB | **~200×** | — |

14B ve 32B'de resident **aynı** (2,63 GB) — footprint sabit. Dağıtımda tensör-shard ile
per-node = toplam/N (N sınırsız).

## Neden AirLLM/Colibri yerine

| | AirLLM | Colibri | IRIS |
|---|---|---|---|
| Kalite | kayıplı olabilir | kayıplı akış | **bit-birebir 0.0** |
| Aynı ayak-izinde | — | 28/48 sapar | **48/48 kesin** |
| Disk okuma/token | 2,62 GB | — | **0,66 GB (7,4× az)** |
| Modern ortamda kurulum | ✗ (deprecated bağımlılık) | derleme | **✓ tek komut** |
| Doğrulanabilirlik | — | — | **iris verify → 0.0** |
| Ölçek | tek makine | tek makine | **tek + N-node HTTP** |

Canlı ölçüldü (aynı Kaggle CPU): IRIS tek komutla kuruldu+koştu (1,72 GB, bit-birebir); AirLLM
modern transformers/optimum'da **kurulmadı**. Herkes `iris verify` ile bit-birebirliği doğrular.

## Test

```bash
python -m iris.tests.test_smoke            # import + CLI + API yüzeyi
iris verify Qwen/Qwen2.5-0.5B              # tam entegrasyon: bit-birebir 0.0
```

## Nasıl çalışır

1. **Böl** — model katman/tensör düzeyinde parçalanır.
2. **Akıt** — her parça gerektiğinde belleğe alınır, kullanılınca bırakılır (disk-offload yok).
3. **Birleştir** — aynı ağırlık + aynı sıra = tam modelle bit-birebir çıktı.

Apache-2.0 · Norovox Labs · PRİZMA çekirdeği
