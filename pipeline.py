"""
MaintNet-tarzi bakim iş emri (MWO) dijitallestirme pipeline'i
==============================================================
Dagdinik, serbest-metin bakim loglarini -> temiz, yapilandirilmis, sorgulanabilir
bir veritabanina cevirir.

Akis:  ingest -> profile (oncesi) -> clean -> structure -> standardize
       -> secure -> load (SQLite) -> evaluate (sonrasi)

NOT: Bu surum tamamen OFFLINE calisir (kural tabanli cikarim).
     Gercek projede `structure_record()` icindeki LLM kancasini acarsin.
"""

import re
import os
import json
import argparse
import sqlite3
import hashlib
from datetime import datetime
from difflib import get_close_matches

import pandas as pd

DATA_DIR = "real_data"
DB_PATH = "output/clean_maintenance.db"

# LLM cikarim ayarlari (use_llm=True veya compare_llm.py kullaninca devreye girer)
# Backend: "ollama" (YEREL, ucretsiz) | "anthropic" (bulut, ucretli API)
LLM_BACKEND = "ollama"
LLM_MODEL = "qwen2.5:3b"               # ollama icin; anthropic icin "claude-opus-4-8"
OLLAMA_HOST = "http://localhost:11434"
_LLM_CLIENT = None

# ---------------------------------------------------------------------------
# 0) KONFIGURASYON (sozlukler)
# ---------------------------------------------------------------------------

# Her alanin KENDI sema yapisi var (heterojen kaynaklar). Ortak semaya esliyoruz.
# Gercek MaintNet verisinin kolon isimleri (people.rit.edu/fa3019/MaintNet).
# Bazi alanlar kaynak veride YOK -> None birakiyoruz; ingest bunu "" olarak doldurur.
#   aviation : tarih ve kisi alani yok (veri zaten deidentified)
#   *        : hicbir kaynakta ayri "person" kolonu yok
SCHEMA_MAP = {
    "aviation": {
        "file": "maintnet_aviation_dataset_deidentified.csv",
        "id": "IDENT", "problem": "PROBLEM", "action": "ACTION",
        "date": None, "person": None,
    },
    "automotive": {
        "file": "Labeled_Car_Dataset200.csv",
        "id": "jobno", "problem": "Notes", "action": "Notes",
        "date": "JobDate", "person": None,
    },
    "facility": {
        "file": "Facility_Maintenance200.csv",
        "id": "WORK_ID", "problem": "DESCRIPTION", "action": "DESCRIPTION",
        "date": "DATE_REQUESTED", "person": None,
    },
}

def _load_abbrev():
    """MaintNet'in gercek kisaltma CSV'lerini birlestirip {abbr: aciklama} sozlugu olusturur."""
    files = [
        "aviation_abbriviation.csv",
        "car_abbriviation.csv",
        "facility_abbriviation.csv",
    ]
    result = {}
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            abbr = str(row.get("Abbreviated", "")).strip().lower()
            desc = str(row.get("Standard_Description", "")).strip().lower()
            if abbr and desc:
                result[abbr] = desc
    return result

ABBREV = _load_abbrev()


def _load_lexicons():
    """MaintNet'in grammar + termBank dosyalarindan VOCAB ve aday ASSET_CANON uretir.

    - grammar.csv / facilty_grammar.csv : Word, Lemma, POS  -> vocab + isim (NN) varliklari
    - domain_words2_termBank.csv / facility_domain.csv : Word -> vocab
    """
    grammar_files = ["grammar.csv", "facilty_grammar.csv"]
    term_files = ["domain_words2_termBank.csv", "facility_domain.csv"]
    vocab = set()
    noun_assets = {}   # variant (yuzey form) -> kanonik (lemma)

    for fname in grammar_files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            word = str(row.get("Word", "")).strip().lower()
            lemma = (str(row.get("Lemma", "")).strip().lower() or word)
            pos = str(row.get("Part of Speech (POS)", "")).strip().upper()
            if word.isalpha():
                vocab.add(word)
            if lemma.isalpha():
                vocab.add(lemma)
            # isimler (NN/NP) -> aday varlik; kanonik = lemma
            if pos.startswith("N") and word.isalpha() and len(word) > 2:
                noun_assets[word] = lemma.replace(" ", "_")

    for fname in term_files:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        for w in df.get("Word", pd.Series([], dtype=str)):
            for tok in str(w).strip().lower().split():
                if tok.isalpha() and len(tok) > 2:
                    vocab.add(tok)

    return vocab, noun_assets


_GRAMMAR_VOCAB, _GRAMMAR_ASSETS = _load_lexicons()

# Yazim duzeltme icin alan sozlugu (difflib referansi)
#   = ABBREV aciklamalari + MaintNet grammar/termBank kelimeleri + sabit takviye set
VOCAB = sorted(set(ABBREV.values()) | _GRAMMAR_VOCAB | {
    "seal", "leak", "leaking", "crack", "cracked", "worn", "loose", "missing",
    "overheating", "cooling", "pressure", "noise", "broken", "stuck", "jammed",
    "slipping", "flickering", "failure", "tire", "strut", "actuator", "valve",
    "cover", "control", "arm", "clutch", "boiler", "elevator", "chiller",
    "ballast", "faucet", "compressor", "filter", "belt", "battery", "windshield",
    "installed", "replaced", "fabricated", "resecured", "serviced", "resealed",
    "cleaned", "lubed", "aligned", "recharged", "refilled", "repaired", "rooftop",
})

# Varlik/komponent -> kanonik isim (standartlastirma + dedup)
#   Elle kurulmus cok-kelimeli kanonikler (oncelikli) + grammar'dan turetilen isimler
_ASSET_CURATED = {
    "valve cover": "valve_cover", "water pump": "water_pump",
    "control arm": "control_arm", "water heater": "water_heater",
    "push rod tube": "push_rod_tube", "hvac": "hvac_unit",
    "ahu": "air_handling_unit", "ac": "ac_unit",
    "baffle": "baffle", "seal": "seal", "gasket": "gasket", "engine": "engine",
    "cylinder": "cylinder", "screw": "screw", "tire": "tire", "brake": "brake",
    "strut": "strut", "actuator": "actuator", "battery": "battery",
    "transmission": "transmission", "compressor": "compressor",
    "exhaust": "exhaust", "clutch": "clutch", "boiler": "boiler",
    "elevator": "elevator", "chiller": "chiller", "ballast": "ballast",
    "faucet": "faucet", "windshield": "windshield", "filter": "filter",
    "belt": "belt",
}
# grammar isimleri taban, elle kurulmus olanlar ustte (cakismada elle kurulmus kazanir)
ASSET_CANON = {**_GRAMMAR_ASSETS, **_ASSET_CURATED}

# Ariza turu anahtar kelimeleri
FAILURE_KW = [
    ("leak", "leak"), ("drip", "leak"), ("crack", "crack"), ("worn", "wear"),
    ("loose", "loose"), ("missing", "missing"), ("overheat", "overheat"),
    ("temp high", "overheat"), ("not cooling", "no_cooling"),
    ("no hot water", "no_function"), ("low", "low_level"), ("noise", "noise"),
    ("knock", "noise"), ("broken", "broken"), ("broke", "broken"),
    ("stuck", "stuck"), ("jammed", "stuck"), ("not holding", "no_charge"),
    ("slipping", "slipping"), ("flickering", "flickering"),
    ("failure", "failure"), ("not closing", "no_function"),
]

# Aksiyon turu anahtar kelimeleri
ACTION_KW = [
    ("replac", "replace"), ("rplcd", "replace"), ("rplac", "replace"),
    ("install", "install"), ("fabricat", "fabricate"), ("resecure", "secure"),
    ("resecured", "secure"), ("serviced", "service"), ("reseal", "reseal"),
    ("clean", "inspect_clean"), ("checked", "inspect_clean"), ("lube", "lubricate"),
    ("freed", "lubricate"), ("align", "align"), ("recharg", "recharge"),
    ("refill", "recharge"), ("bled", "recharge"), ("flush", "flush"),
    ("repair", "repair"), ("cleared", "repair"),
]

DATE_FORMATS = [
    "%m/%d/%y %H:%M",   # facility:   "1/25/19 0:00"
    "%m/%d/%y",         # automotive: "1/6/15"
    "%m/%d/%Y", "%Y-%m-%d", "%d.%m.%Y", "%B %d %Y", "%b %d %Y",
]


# ---------------------------------------------------------------------------
# 1) INGEST -- 3 heterojen CSV'yi tek ortak semaya
# ---------------------------------------------------------------------------
def ingest():
    frames = []
    for domain, cols in SCHEMA_MAP.items():
        # utf-8-sig: aviation dosyasindaki BOM'u temizler
        df = pd.read_csv(f"{DATA_DIR}/{cols['file']}",
                         dtype=str, encoding="utf-8-sig").fillna("")

        def col(key):
            """Kaynak veride olmayan alanlar (None) -> bos string."""
            name = cols.get(key)
            return df[name] if name and name in df.columns else ""

        out = pd.DataFrame({
            "record_id":   col("id"),
            "domain":      domain,
            "problem_raw": col("problem"),
            "action_raw":  col("action"),
            "date_raw":    col("date"),
            "person_raw":  col("person"),
        })
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 2) PROFILE -- "oncesi" kalite fotografi
# ---------------------------------------------------------------------------
def profile(df, label):
    total = len(df)
    miss_action = (df["action_raw"].str.strip() == "").sum()
    miss_date = (df["date_raw"].str.strip() == "").sum()
    # kisaltma/jargon iceren kayit orani
    abbr_keys = set(ABBREV.keys())
    def has_abbr(t):
        toks = re.findall(r"[a-z/]+", str(t).lower())
        return any(tok in abbr_keys for tok in toks)
    abbr_rows = df["problem_raw"].apply(has_abbr).sum()
    date_formats = df["date_raw"].apply(lambda s: _date_signature(s)).nunique()
    print(f"\n=== PROFIL [{label}] ===")
    print(f"  Kayit sayisi          : {total}")
    print(f"  Eksik aksiyon         : {miss_action}  (%{100*miss_action/total:.0f})")
    print(f"  Eksik tarih           : {miss_date}  (%{100*miss_date/total:.0f})")
    print(f"  Kisaltma/jargon iceren: {abbr_rows}  (%{100*abbr_rows/total:.0f})")
    print(f"  Farkli tarih formati  : {date_formats}")
    return {"total": total, "miss_action": int(miss_action),
            "miss_date": int(miss_date), "abbr_rows": int(abbr_rows),
            "date_formats": int(date_formats)}

def _date_signature(s):
    s = str(s).strip()
    if not s:
        return "empty"
    if re.match(r"\d{4}-\d{2}-\d{2}", s): return "iso"
    if re.match(r"\d{1,2}/\d{1,2}/\d{4}", s): return "slash"
    if re.match(r"\d{1,2}\.\d{1,2}\.\d{4}", s): return "dot"
    return "text"


# ---------------------------------------------------------------------------
# 3) CLEAN -- kisaltma acma + yazim duzeltme + tarih normalizasyonu
# ---------------------------------------------------------------------------
def expand_and_correct(text):
    if not text:
        return ""
    out = []
    for tok in re.findall(r"[a-z/#0-9]+", text.lower()):
        if tok in ABBREV:                       # bilinen kisaltma
            out.append(ABBREV[tok])
        elif tok.isalpha() and len(tok) > 3 and tok not in VOCAB:
            m = get_close_matches(tok, VOCAB, n=1, cutoff=0.82)  # yazim duzeltme
            out.append(m[0] if m else tok)
        else:
            out.append(tok)
    return " ".join(out)

def normalize_date(s):
    s = str(s).strip()
    if not s:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# 4) STRUCTURE -- serbest metin -> {asset, failure_mode, action_type}
# ---------------------------------------------------------------------------
def structure_record(problem_clean, action_clean, use_llm=False):
    if use_llm:
        return llm_extract(problem_clean, action_clean)   # gercek projede burasi
    return rule_based_extract(problem_clean, action_clean)

# Uzun (cok-kelimeli) varyantlari once dene -> "valve cover" "valve"den once eslessin
_ASSET_VARIANTS = sorted(ASSET_CANON, key=lambda v: -len(v))

def rule_based_extract(problem, action):
    asset = None
    for variant in _ASSET_VARIANTS:
        if variant in problem or variant in action:
            asset = ASSET_CANON[variant]
            break
    failure = next((cat for kw, cat in FAILURE_KW if kw in problem), None)
    act = next((cat for kw, cat in ACTION_KW if kw in action), None)
    return {"asset": asset, "failure_mode": failure, "action_type": act}

# LLM'in dondurecegi yapilandirilmis cikti semasi (gecerli JSON garantisi)
_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "asset": {"type": "string",
                  "description": "Ana ekipman/komponent, snake_case kanonik (or. water_pump). Yoksa 'none'."},
        "failure_mode": {"type": "string",
                         "description": "Ariza turu kategorisi (or. leak, crack, overheat). Yoksa 'none'."},
        "action_type": {"type": "string",
                        "description": "Yapilan islem kategorisi (or. replace, repair, inspect). Yoksa 'none'."},
    },
    "required": ["asset", "failure_mode", "action_type"],
    "additionalProperties": False,
}

# Veri Ingilizce oldugu icin istem de Ingilizce -> kucuk modellerde daha tutarli
_LLM_SYSTEM = (
    "You structure industrial maintenance log entries. "
    "From the given problem and action text, extract three fields: "
    "asset (main equipment/component, snake_case canonical), "
    "failure_mode (the fault category, e.g. leak, crack, overheat, wear), "
    "action_type (the maintenance action, e.g. replace, repair, inspect, lubricate). "
    "If a field is not present in the text, output 'none'. "
    "Return only JSON matching the requested schema."
)


def _get_llm_client():
    """LLM istemcisini tembel (lazy) olusturur -- secili backend'e gore."""
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        if LLM_BACKEND == "ollama":
            import ollama
            _LLM_CLIENT = ollama.Client(host=OLLAMA_HOST)
        elif LLM_BACKEND == "anthropic":
            import anthropic
            _LLM_CLIENT = anthropic.Anthropic()   # ANTHROPIC_API_KEY ortamdan
        else:
            raise ValueError(f"Bilinmeyen LLM_BACKEND: {LLM_BACKEND!r}")
    return _LLM_CLIENT


def _norm_field(v):
    v = str(v).strip().lower()
    return None if v in ("", "none", "null", "yok", "n/a") else v


def llm_extract(problem, action):
    """Tek bir kaydi LLM ile yapilandirir -> {asset, failure_mode, action_type}.

    Kural tabanli yontemden farki: sozlukte OLMAYAN kisaltma/yazim hatasini da cozer,
    baglamdan cikarim yapar. Yeni alan eklemek = sadece semayi/istemi degistirmek.
    Her iki backend de JSON-sema ile YAPILANDIRILMIS cikti uretir (gecerli JSON garantisi).
    """
    client = _get_llm_client()
    prompt = f'Sorun: "{problem}"\nAksiyon: "{action}"'

    if LLM_BACKEND == "ollama":
        resp = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": _LLM_SYSTEM},
                      {"role": "user", "content": prompt}],
            format=_LLM_SCHEMA,                 # JSON sema -> yapilandirilmis cikti
            options={"temperature": 0},
        )
        text = resp["message"]["content"]
    else:  # anthropic
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=200,
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": _LLM_SCHEMA}},
        )
        text = next(b.text for b in resp.content if b.type == "text")

    data = json.loads(text)
    return {"asset": _norm_field(data.get("asset")),
            "failure_mode": _norm_field(data.get("failure_mode")),
            "action_type": _norm_field(data.get("action_type"))}


# ---------------------------------------------------------------------------
# 5) SECURE -- basit PII tespiti + maskeleme
# ---------------------------------------------------------------------------
def mask_person(value):
    v = str(value).strip()
    if not v:
        return None
    # kararli takma kimlik (geri donulemez hash)
    h = hashlib.sha1(v.encode()).hexdigest()[:8]
    return f"PERSON_{h}"


# ---------------------------------------------------------------------------
# 6) Kalite skoru
# ---------------------------------------------------------------------------
def quality_score(rec):
    fields = [rec["asset"], rec["failure_mode"], rec["action_type"], rec["date"]]
    return round(sum(f is not None for f in fields) / len(fields), 2)


# ---------------------------------------------------------------------------
# 7) LOAD -- temiz yapilandirilmis kayitlari SQLite'a yaz
# ---------------------------------------------------------------------------
def load_to_db(records):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS maintenance")
    conn.execute("""
        CREATE TABLE maintenance (
            record_id TEXT, domain TEXT, asset TEXT, failure_mode TEXT,
            action_type TEXT, date TEXT, person_id TEXT,
            problem_clean TEXT, quality REAL)
    """)
    conn.executemany(
        "INSERT INTO maintenance VALUES (?,?,?,?,?,?,?,?,?)",
        [(r["record_id"], r["domain"], r["asset"], r["failure_mode"],
          r["action_type"], r["date"], r["person_id"], r["problem_clean"],
          r["quality"]) for r in records])
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# LLM cache -- ayni metni iki kez LLM'e gondermemek icin kalici onbellek
# ---------------------------------------------------------------------------
LLM_CACHE_PATH = "output/llm_cache.json"
CHECKPOINT_EVERY = 500          # her bu kadar kayitta bir DB+CSV+cache snapshot'i al

def _cache_key(problem, action):
    # model adi da anahtara dahil -> model degisince cache karismaz
    return hashlib.sha1(f"{LLM_MODEL}||{problem}||{action}".encode()).hexdigest()

def _load_llm_cache():
    if os.path.exists(LLM_CACHE_PATH):
        with open(LLM_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_llm_cache(cache):
    os.makedirs(os.path.dirname(LLM_CACHE_PATH), exist_ok=True)
    with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def _stratified_sample(df, limit, seed=42):
    """Domain-dengeli ornek (or. limit=600 -> her domainden ~200)."""
    per = max(1, limit // df["domain"].nunique())
    parts = [g.sample(min(len(g), per), random_state=seed)
             for _, g in df.groupby("domain")]
    return pd.concat(parts).reset_index(drop=True)


def _write_outputs(records):
    """O ana kadarki kayitlari hem SQLite hem CSV'ye yazar (checkpoint + final)."""
    load_to_db(records)
    pd.DataFrame(records).to_csv("output/clean_maintenance.csv", index=False)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run(use_llm=True, limit=None):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    df = ingest()
    if limit and limit < len(df):
        df = _stratified_sample(df, limit)
    before = profile(df, "HAM VERI (oncesi)")

    cache = _load_llm_cache() if use_llm else {}
    n = len(df)
    if use_llm:
        print(f"\nLLM cikarim: {LLM_BACKEND}/{LLM_MODEL}  ({n} kayit, cache={len(cache)})")
        print(f"Checkpoint: her {CHECKPOINT_EVERY} kayitta bir DB+CSV+cache yazilir.")
        print("Kesilirse tekrar calistir -> cache'deki kayitlar aninda atlanir, kaldigi yerden devam.\n")

    records = []
    new_calls = 0          # bu calistirmada yapilan YENI LLM cagrisi sayisi
    for i, (_, row) in enumerate(df.iterrows(), 1):
        problem_clean = expand_and_correct(row["problem_raw"])
        action_clean = expand_and_correct(row["action_raw"])

        if use_llm:
            key = _cache_key(problem_clean, action_clean)
            if key in cache:
                fields = cache[key]                      # onbellekten -> aninda
            else:
                try:
                    fields = llm_extract(problem_clean, action_clean)
                    cache[key] = fields                  # sadece basariyi cache'le
                    new_calls += 1
                except Exception:
                    fields = rule_based_extract(problem_clean, action_clean)  # yedek
        else:
            fields = rule_based_extract(problem_clean, action_clean)

        rec = {
            "record_id": row["record_id"],
            "domain": row["domain"],
            "problem_clean": problem_clean,
            "date": normalize_date(row["date_raw"]),
            "person_id": mask_person(row["person_raw"]),
            **fields,
        }
        rec["quality"] = quality_score(rec)
        records.append(rec)

        # --- CHECKPOINT: her CHECKPOINT_EVERY kayitta cache + DB + CSV snapshot'i ---
        if use_llm and i % CHECKPOINT_EVERY == 0:
            _save_llm_cache(cache)
            _write_outputs(records)
            print(f"  [checkpoint] {i}/{n} islendi  |  yeni LLM cagrisi: {new_calls}  "
                  f"|  cache: {len(cache)}  |  DB+CSV yazildi")

    # final yazim
    if use_llm:
        _save_llm_cache(cache)
    _write_outputs(records)

    # "sonrasi" degerlendirme
    out = pd.DataFrame(records)
    print("\n=== SONUC (sonrasi) ===")
    print(f"  Kayit sayisi              : {len(out)}")
    print(f"  Asset cikarilan           : %{100*out['asset'].notna().mean():.0f}")
    print(f"  Failure_mode cikarilan    : %{100*out['failure_mode'].notna().mean():.0f}")
    print(f"  Action_type cikarilan     : %{100*out['action_type'].notna().mean():.0f}")
    print(f"  Tarih normalize edilen    : %{100*out['date'].notna().mean():.0f}")
    print(f"  Benzersiz kanonik asset   : {out['asset'].nunique()}")
    print(f"  Ortalama kalite skoru     : {out['quality'].mean():.2f}")
    masked = out['person_id'].notna().sum()
    if masked:
        print(f"  PII maskelendi            : {masked} kisi alani -> PERSON_<hash>")
    else:
        print(f"  PII maskelendi            : kaynak veride kisi alani yok (deidentified)")

    if use_llm:
        print(f"  Bu calistirmada yeni LLM cagrisi: {new_calls}  (gerisi cache'den)")

    print("\n--- Ornek temiz kayitlar ---")
    print(out[["domain", "asset", "failure_mode", "action_type", "date", "quality"]]
          .head(8).to_string(index=False))

    print(f"\nCikti: {DB_PATH}  ve  output/clean_maintenance.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MaintNet temizleme + LLM yapilandirma hatti")
    ap.add_argument("--limit", type=int, default=None,
                    help="domain-dengeli ornek (or. 600). Bos: tum 6569 kayit")
    ap.add_argument("--rule", action="store_true",
                    help="LLM yerine kural-tabanli cikarim kullan")
    args = ap.parse_args()
    run(use_llm=not args.rule, limit=args.limit)
