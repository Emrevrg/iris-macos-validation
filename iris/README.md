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
Model bitleri hiç atılmaz; ağırlıklar katman katman / dilim dilim akıtılır, kullanılınca
bırakılır. Çıktı, tam-yüklü referansla logit düzeyinde **birebir aynı** — kendin
`iris verify` ile doğrularsın.

> Bit **sıkıştırma** değil (yüksek-entropi ağırlıkların kayıpsız sınırı ~1,3× — fizik).
> Kayıpsız **akış/dağıtım**: yerleşik ayak-izi ≈ aktif dilim, model boyutundan bağımsız.

## Kur

```bash
pip install git+https://huggingface.co/emrevrg/iris-engine
```

## Kullan

```bash
iris run Qwen/Qwen2.5-7B --prompt "Görelilik nedir?" --max-new 40     # KV-cache'li üretim
iris verify Qwen/Qwen2.5-7B                                          # 4 istem, tüm pozisyonlar + üretim
iris verify Qwen/Qwen2.5-7B --low-mem --tensor-shards 8              # dilim-akışı da bit-birebir mi?
iris bench Qwen/Qwen2.5-7B                                           # ilk-token, decode tok/s, GB/token

# diske YAZMADAN, doğrudan Hub'dan (HTTP Range) — disk modelden küçük olsa bile
iris run Qwen/Qwen2.5-72B --source hub --low-mem --tensor-shards 32

# veri-merkezi modu: ağırlıklar pinned host RAM'de, GPU'da yalnız aktif katman
iris run Qwen/Qwen2.5-3B --source ram --device cuda
```

| bayrak | ne yapar |
|---|---|
| `--low-mem` | embed/lm_head yerleşik tutulmaz, satır/blok olarak akıtılır |
| `--tensor-shards N` | her Linear N satır-dilimine bölünür (kolon-paralel); yerleşik ağırlık ≈ katman/N |
| `--source local\|hub\|ram` | yerel safetensors (mmap) · Hub'dan Range akışı · pinned host RAM |
| `--chat / --no-chat` | varsayılan: şablon yalnız instruct/chat modellerinde |

Koddan:

```python
from iris.engine import IrisModel, verify_bit_exact
m = IrisModel("Qwen/Qwen2.5-7B", low_mem=True, tensor_shards=8)
print(m.generate("The theory of relativity", max_new_tokens=20))
print(m.last_stats)          # ilk-token, decode tok/s, okunan GB, tepe RSS
verify_bit_exact("Qwen/Qwen2.5-7B", tensor_shards=8)

# akış üzerinden ince ayar: taban dondurulmuş ve akışta, LoRA yerleşik
params = m.add_lora(rank=8, targets=("q_proj", "v_proj"))
loss = m.loss(ids); loss.backward()
```

## Ölçülen kanıtlar (hepsi bulutta koşuldu, JSON artefaktlı)

**Bit-kesinlik** (Kaggle CPU; 4 istem × tüm pozisyonlar + KV-cache'li 8 adım üretim; referans =
`transformers.from_pretrained` tam model):

| model | mod | karşılaştırılan logit | max \|Δ\| | üretim tokenları |
|---|---|---|---|---|
| Qwen2.5-0.5B | normal | 11,2 M | **0.0** | birebir |
| Qwen2.5-0.5B | low_mem | 11,2 M | **0.0** | birebir |
| Qwen2.5-0.5B | 8 tensör-shard | 11,2 M | **0.0** | birebir |
| Qwen2.5-0.5B | low_mem + 8 shard | 11,2 M | **0.0** | birebir |
| Qwen2.5-0.5B | **Hub'dan diske yazmadan** + 4 shard | 11,2 M | **0.0** | birebir |
| Qwen2.5-0.5B-Instruct | normal | 11,2 M | **0.0** | birebir |
| Qwen2.5-7B | normal | 3,3 M | **0.0** | birebir |
| Qwen2.5-7B | low_mem + 8 shard | 3,3 M | **0.0** | birebir |

**Görülmemiş istemde GPU kesin gecikme** (Tesla T4, Qwen2.5-3B, istem koşu anında
`os.urandom` ile üretildi; referans = tam model aynı GPU'da `model.generate`): 16/16 adımda
logit farkı **0.0**, tokenlar birebir. Adım gecikmesi p50 **0,54 s**, p95 **0,69 s**; GPU'da yerleşik
**0,81 GB** (model 6,17 GB). Token başına tüm model PCIe'den akar — ~12 GB/s, veriyolu sınırında.

**Hız (KV-cache'siz ilk yol → KV-cache'li yol)**: 0.5B'de **2,4×**, 7B'de (12 token) **4,3×** hızlanma. 7B CPU'da
IRIS 3,3 GB RAM ile 26,6 s; tam model (17 GB RAM) 16,3 s.

**Akış üzerinden ince ayar + optimizer kaldığı-yerden-devam** (Qwen2.5-0.5B, LoRA r=8, 3 ayrı
süreç): 6. adımda kes → kaydet → yeni süreçte devam; kesintisiz koşuyla LoRA parametre farkı
**0.0**, AdamW durum farkı **0.0**, 12 kayıp değeri birebir; taban safetensors SHA-256 değişmedi.

## Ölçekleme (ölçülen)

| Model | Boyut | Yerleşik | Küçültme | Bit-birebir |
|---|---|---|---|---|
| Qwen2.5-7B | 15,2 GB | 1,78 GB RAM | 8,5× | ✓ 0.0 |
| Qwen2.5-14B | 29,5 GB | 2,63 GB RAM | 11,2× | ✓ |
| Qwen2.5-32B | 65,5 GB | 2,63 GB RAM | 24,9× | ✓ |
| Qwen2.5-14B, **T4 GPU** (15,6 GB) | 29,5 GB | **0,72 GB VRAM** | **41×** | ✓ 0.0 (referans aynı makinenin 2×T4'ünde) |
| **Qwen2.5-72B, diske yazmadan Hub'dan**, 32 dilim | 145,4 GB | **1,71 GB RAM** | **85×** | ✓ 0.0 (dilim ↔ katman modu, 912 K logit) |

**Veri merkezi kapasitesi** (aynı Kaggle makinesi, 2×T4): IRIS 29,5 GB'lık 14B'yi **tek** T4'te
0,72 GB tepe VRAM ile koşar; standart `transformers` referansı modeli tutmak için **iki** T4'ü de
kullanır. 8 adım greedy üretim ve 70 token kalite metninde logit farkı **0.0**, perplexity birebir
(3,1157). Hız dürüstçe: IRIS 0,0097 tok/s (her token 29,5 GB diskten akar), referans 1,47 tok/s.

## Kayıpsız fiziksel paket (61 GiB, kaynaksız ve ağsız geri yükleme)

Qwen2.5-32B (commit sabit, 27 dosya, 65,5 GB) Hub'dan **akıtılarak** (kaynak hiç diske yazılmadan)
bf16 bayt-düzlemi + zstd ile paketlendi: **46,7 GB (0,713×)**, 25 volume, 4 ayrı kernel'ın kalıcı
çıktısında. Geri yükleme **interneti kapalı** ayrı bir makinede (farklı ağ ad-alanı, prob engelli):
**27/27 dosya SHA-256 birebir** (Hub LFS özetleriyle de), 25 volume SHA doğrulandı, bit-çevrilmiş
volume hem volume SHA'sında hem parça SHA'sında **reddedildi**.

## Platformlar (ölçülen)

| platform | kanıt |
|---|---|
| **Linux** (Kaggle x86_64, CPU + T4 GPU) | tüm modlar bit-birebir (yukarıdaki tablolar) |
| **Windows 10** (x86_64, CPU) | normal · low_mem · 4 shard · low_mem+4 shard · Hub akışı → 5/5 **0.0**, her biri 11,2 M logit |
| macOS | CI hazır; GitHub hesabı ödeme kilidi nedeniyle koşturulamadı |

Katı kabul denetimi (`scripts/iris_acceptance.py`, fail-closed): **6/7** — bütünlük, fiziksel
paket, görülmemiş istemde kesin gecikme, GGUF/MoE/multimodal uyumluluk, ince ayar + devam,
veri-merkezi kapasitesi GEÇTİ; yalnız macOS ölçümü eksik.

## Neden AirLLM/Colibri yerine

| | AirLLM | Colibri | IRIS |
|---|---|---|---|
| Kalite | kayıplı olabilir | kayıplı akış | **bit-birebir 0.0** |
| KV-cache'li akış üretimi | — | — | **✓ (4,3× hızlı)** |
| Tensör-dilim akışı | — | — | **✓ N dilim, bit-birebir** |
| Diske yazmadan Hub akışı | — | — | **✓** |
| Akış üzerinden ince ayar + devam | — | — | **✓ bit-birebir devam** |
| Modern ortamda kurulum | ✗ (deprecated bağımlılık) | derleme | **✓ tek komut** |
| Doğrulanabilirlik | — | — | **iris verify → 0.0** |

## Test

```bash
python -m iris.tests.test_smoke     # import + CLI + API yüzeyi + streamer
iris verify Qwen/Qwen2.5-0.5B       # tam entegrasyon: bit-birebir 0.0
```

## Nasıl çalışır

1. **Böl** — model katman ve tensör-satır dilimlerine ayrılır (safetensors başlığından, yüklemeden).
2. **Akıt** — her dilim gerektiği an okunur (disk mmap / Hub Range / pinned RAM), bir sonraki
   arka planda önden getirilir, kullanılınca bırakılır.
3. **Birleştir** — aynı ağırlık + aynı çekirdek şekli = tam modelle bit-birebir. GPU'da
   lm_head'in şekli de referansla eşlenir (`logits_to_keep=1`), aksi hâlde cuBLAS farklı
   algoritma seçer (ölçüldü: 0,0156 → 0.0).

Sınır (dürüst): bit-kesinlik aynı cihaz/dtype/kütüphane sürümünde garanti edilir; farklı
makineler arası float farkları IRIS'e değil donanıma aittir.

Apache-2.0 · Norovox Labs · PRİZMA çekirdeği
