# AI for Data Management — MaintNet Bakım Logu Dijitalleştirme

Fabrikalardaki **düzensiz, serbest-metin bakım loglarını** AI yardımıyla temizleyip
yapılandırılmış, sorgulanabilir bir veritabanına aktaran uçtan uca bir hat.

> Akademik çerçeve: *Intelligent automation for organizing, cleaning, and securing
> corporate databases.* Anomaly detection değil — **veri dijitalleştirme ve düzenleme**.

**Kaynak:** MaintNet (akademik açık kaynak dataset)
- Paper: https://arxiv.org/abs/2005.12443
- Veri barınağı: https://people.rit.edu/fa3019/MaintNet/  (indirme: `autonlab/pmx_data` reposu)

---

## Klasör Yapısı

```
bitirme/
├── pipeline.py              ← Ana hat (ingest → clean → LLM structure → secure → load)
├── compare_llm.py           ← Kural-tabanlı vs LLM kıyas hattı (+ grafik)
├── dashboard.py             ← Streamlit interaktif veri paneli
├── README.md                ← Bu dosya
├── real_data/               ← GERÇEK MaintNet verisi (indirilmiş)
│   ├── maintnet_aviation_dataset_deidentified.csv   (6169 bakım kaydı)
│   ├── Labeled_Car_Dataset200.csv                   (200 otomotiv kaydı)
│   ├── Facility_Maintenance200.csv                  (200 tesis kaydı)
│   ├── *_abbriviation.csv        (3 domain — kısaltma sözlükleri)
│   ├── grammar.csv / facilty_grammar.csv  (POS + lemma)
│   └── domain_words2_termBank.csv / facility_domain.csv  (alan terimleri)
└── output/
    ├── clean_maintenance.db    ← Pipeline çıktısı (SQLite)
    ├── clean_maintenance.csv   ← Pipeline çıktısı (CSV)
    ├── llm_cache.json          ← LLM çıkarım önbelleği (tekrar çalıştırmalar hızlı)
    ├── llm_vs_rule.csv/.png    ← Kıyas çıktıları
    └── gold_template.csv       ← (ops.) elle etiketleme şablonu
```

---

## Pipeline Nedir, Ne Yapar

`pipeline.py` 8 aşamalı bir hat. Her aşama bir veri-yönetimi sorununu çözer:

| Aşama | Fonksiyon | Ne yapar |
|-------|-----------|----------|
| 1 | `ingest()` | 3 farklı yapıdaki CSV'yi (farklı kolon isimleri) ortak şemaya çeker |
| 2 | `profile()` | Ham verinin "öncesi" kalite fotoğrafını çıkarır (eksik %, kısaltma %, tarih çeşitliliği) |
| 3 | `expand_and_correct()` | Kısaltma açma (gerçek MaintNet sözlükleri) + yazım düzeltme (difflib) |
| 4 | `normalize_date()` | Farklı tarih formatlarını ISO 8601'e çevirir |
| 5 | `structure_record()` | Serbest metin → {asset, failure_mode, action_type} — **LLM ile** |
| 6 | `mask_person()` | Kişi alanlarını SHA-1 hash'e çevirir (PII maskeleme) |
| 7 | `load_to_db()` | Temiz kayıtları SQLite'a yazar + kalite skoru ekler |
| 8 | `run()` raporu | "Sonrası" metrikler: alan çıkarım oranları, ortalama kalite skoru |

### Proje başlığıyla eşleşme
- **organizing** → ingest + structure + standardize
- **cleaning** → expand_and_correct + normalize_date
- **securing** → PII maskeleme + kalite skoru

---

## Sözlükler — gerçek MaintNet kaynaklarından

Çıkarım sözlükleri elle yazılmış değil, MaintNet'in kendi dil kaynaklarından **otomatik** yüklenir:
- `ABBREV` (~127): 3 domain abbreviation CSV'sinden
- `VOCAB` (~335): grammar + termBank dosyalarından + takviye
- `ASSET_CANON` (~70): grammar dosyalarındaki isimlerden (POS=NN) + elle kurulmuş çok-kelimeliler

---

## LLM Çıkarımı (yerel, ücretsiz)

5. aşama (`structure_record`) artık **LLM** ile çalışır. Varsayılan backend **Ollama** (yerel, ücretsiz, API anahtarı yok):

```python
# pipeline.py
LLM_BACKEND = "ollama"          # "ollama" (yerel) | "anthropic" (bulut, ücretli)
LLM_MODEL   = "qwen2.5:3b"
```

- Her iki backend de **JSON-şema** ile yapılandırılmış çıktı üretir (geçerli JSON garantisi).
- **Önbellek:** Aynı metin iki kez LLM'e gönderilmez (`output/llm_cache.json`). Tekrar
  çalıştırmalar anında, kısmi ilerleme kaybolmaz.
- LLM hata verirse sessizce kural-tabanlı çıkarıma düşer (sağlamlık).

### Ollama kurulumu (tek seferlik)
```powershell
winget install Ollama.Ollama        # uygulama
pip install ollama                  # python istemcisi
ollama pull qwen2.5:3b              # model (~2GB)
```

---

## Kurulum

```bash
pip install pandas ollama streamlit plotly
# (Anthropic backend için ek: pip install anthropic + ANTHROPIC_API_KEY)
```

Python 3.12. `sqlite3`, `difflib`, `hashlib`, `json`, `re` standart kütüphane.

---

## Çalıştırma

### 1) Pipeline (temizle + yapılandır + yükle)
```bash
python pipeline.py                 # TÜM 6569 kaydı LLM ile işle
python pipeline.py --limit 600     # domain-dengeli örnek (hızlı test)
python pipeline.py --rule          # LLM yerine kural-tabanlı çıkarım
```
Çıktı: `output/clean_maintenance.db` + `.csv`

**Checkpoint & devam etme:** Tam çalıştırma her **500 kayıtta bir** DB+CSV+cache snapshot'ı
alır (`CHECKPOINT_EVERY`). İşlem yarıda kesilirse (Ctrl+C, kapanma) **tekrar
`python pipeline.py` çalıştırman yeterli** — cache'deki kayıtlar anında atlanır, kaldığı
yerden devam eder. İlk tam tur yerel modelde ~1–1.5 saat; sonraki turlar saniyeler.

> Ön koşul: Ollama çalışıyor olmalı. Bağlantı hatası alırsan Ollama uygulamasını aç
> veya `ollama serve` çalıştır.

### 2) Dashboard (interaktif panel)
```bash
streamlit run dashboard.py
```
Tarayıcıda `localhost:8501`. Domain/kalite filtreleri, asset/failure/action dağılımları,
zaman serisi, kalite histogramı, veri tablosu + CSV indirme.

### 3) Kural-tabanlı vs LLM kıyası (opsiyonel rapor)
```bash
python compare_llm.py --n 80               # 80 kayıt kıyas
python compare_llm.py --n 100 --domain aviation
python compare_llm.py --make-gold 40       # gerçek accuracy için etiketleme şablonu
```
Coverage (kapsama) + Agreement (tam/gevşek uyum) ölçer, `output/llm_vs_rule.png` üretir.

---

## Veri Şemaları

### Ortak şema (ingest sonrası)
```python
{record_id, domain, problem_raw, action_raw, date_raw, person_raw}
```

### Temiz şema (pipeline çıktısı)
```python
{
  record_id, domain,
  problem_clean,        # kısaltmalar açılmış, yazım düzeltilmiş
  asset,                # çıkarılan ekipman/varlık
  failure_mode,         # arıza türü kategorisi
  action_type,          # yapılan işlem kategorisi
  date,                 # ISO 8601 (YYYY-MM-DD)
  person_id,            # PERSON_<sha1>  (kaynak veride kişi alanı yoksa null)
  quality,              # 0.0–1.0 kalite skoru
}
```

> Not: Aviation seti zaten deidentified gelir (tarih ve kişi alanı yoktur); otomotiv/tesis
> setlerinde tarih vardır. Pipeline bu heterojenliği eksik-kolona dayanıklı şekilde yönetir.

---

## Notlar / Bilinen Sınırlar
- **Asset over-specification:** Küçük yerel model asset'i çok özgül üretebilir
  (`intake-gasket`); dashboard çekirdek ismi (`gasket`) gruplayarak gösterir.
- Gerçek **accuracy** için elle etiketli gold kümesi gerekir (`compare_llm.py --make-gold`).
  Aksi halde kıyas "kapsama + uyum" ölçer.
- Tüm 6569 kaydın ilk LLM işlemesi yerel modelde ~1.5 saat sürer; sonraki çalıştırmalar
  önbellekten anında gelir.
