# AI for Data Management — Digitizing Free-Text Maintenance Logs

An end-to-end pipeline that turns **messy, free-text maintenance logs** into a **clean,
structured, queryable database** using a small **Large Language Model (LLM) running
locally at zero cost**.

> Academic framing: *intelligent automation for **organizing**, **cleaning** and
> **securing** corporate data.* The focus is **data digitization and organization** —
> not anomaly or failure prediction.

**Data source:** [MaintNet](https://arxiv.org/abs/2005.12443) — an open-access
maintenance-language dataset (aviation, automotive and facility domains).

---

## What it does

Technicians record every repair as a short free-text note, dense with abbreviations,
typos, inconsistent dates and missing fields — so the logs cannot be searched or
analysed. This project ingests **6,569 real records from 3 heterogeneous domains**,
cleans them, structures each note into explicit fields with an LLM, masks personal data,
and loads everything into SQLite. On top of the clean data it provides an interactive
dashboard and a retrieval-augmented maintenance assistant.

| Raw note | → | Clean structured record |
|----------|---|--------------------------|
| `"#2 intake gasket leaking; rplcd r/h engine"` | → | `asset=gasket · failure_mode=leak · action_type=replace · quality=0.75` |

---

## Repository layout

```
.
├── pipeline.py        # main pipeline: ingest → clean → structure (LLM) → secure → load
├── advisor.py         # retrieval-augmented maintenance assistant (problem → recommendation)
├── compare_llm.py     # rule-based vs LLM comparison (coverage + agreement)
├── evaluate.py        # accuracy vs a hand-labeled gold set (precision / recall / F1)
├── make_gold.py       # builds the 60-record hand-labeled gold test set
├── pii.py             # in-text PII detection & masking (NER + regex)
├── scan_pii.py        # scans the whole corpus for PII, writes a summary
├── dashboard.py       # Streamlit dashboard (5 tabs)
├── requirements.txt
└── real_data/         # MaintNet source CSVs + abbreviation / grammar / term-bank files
```

Generated artifacts (the SQLite DB, CSVs, LLM cache, charts) are written to `output/`
and are **not** tracked in git.

---

## The pipeline (`pipeline.py`)

Eight stages, each solving one data-management problem:

| # | Function | What it does | Pillar |
|---|----------|--------------|--------|
| 1 | `ingest` | Map 3 heterogeneous CSVs into one common schema | Organizing |
| 2 | `profile` | Capture the "before" quality snapshot | — |
| 3 | `expand_and_correct` | Expand abbreviations, correct spelling | Cleaning |
| 4 | `normalize_date` | Convert dates to ISO 8601 | Cleaning |
| 5 | `structure_record` | Free text → `{asset, failure_mode, action_type}` via a local LLM | Organizing |
| 6 | `mask_person` | Hash personally identifiable information | Securing |
| 7 | `load_to_db` | Write clean records to SQLite + quality score | Securing |
| 8 | `run` report | Capture the "after" metrics | — |

**Cleaning dictionaries are loaded automatically** from MaintNet's own resources
(`ABBREV` ≈ 127, `VOCAB` ≈ 335, `ASSET_CANON` ≈ 70) — they are not hand-written.

### Local LLM (free, no API key)

```python
# pipeline.py
LLM_BACKEND = "ollama"      # "ollama" (local) | "anthropic" (cloud, paid)
LLM_MODEL   = "qwen2.5:3b"
```

Both backends produce **schema-constrained JSON** (valid output guaranteed). A persistent
**cache** (`output/llm_cache.json`) means the same text is never sent twice, and the run
**checkpoints every 500 records** — so a long run is resumable, and if the LLM fails on a
record it falls back to rule-based extraction.

---

## Evaluation

- **`compare_llm.py`** — rule-based vs LLM, measuring **coverage** (% filled) and
  **agreement** (do the two methods produce the same value).
- **`make_gold.py` + `evaluate.py`** — a **60-record hand-labeled gold set** (20 per
  domain) enables true **accuracy, precision, recall and F1**, plus per-domain accuracy
  and an error analysis of where the LLM disagrees with the ground truth.

**Headline results (full corpus / gold set):**

| Metric | Value |
|--------|-------|
| Asset / action / failure extracted | 99% / 99% / 90% |
| Average quality score | 0.73 |
| Asset F1 (LLM, gold set) | **0.73** (vs 0.33 rule-based) |
| Macro-F1 (LLM vs rule-based) | **66% vs 56%** |

---

## Securing — in-text PII masking (`pii.py`, `scan_pii.py`)

Real personal data hides inside the notes themselves. The pipeline detects **person
names (spaCy NER, gated by trigger context for precision)**, **phone numbers and e-mails
(regex)**, and masks them. On this corpus it finds **15 records with PII (11 names,
7 phone numbers), all in the facility domain** — the aviation set is de-identified, so
the detector correctly flags nothing there. Raw PII values are never stored.

---

## Maintenance assistant (`advisor.py`)

Given a free-text problem, the assistant (1) structures it with the LLM, (2) retrieves
similar past cases from the clean database, (3) recommends the most common action taken,
and (4) writes short advice grounded only in those real past cases (retrieval-augmented
generation). This would be impossible over the raw, unstructured logs.

---

## Dashboard (`dashboard.py`)

An interactive Streamlit app with five tabs:

- **Overview** — KPIs and charts over the whole dataset.
- **Maintenance Records** — filter & free-text search; raw note beside the AI-extracted fields.
- **Before / After** — raw messy data vs clean structured data, incl. the PII-masking panel.
- **Assistant** — type a problem, get a recommendation.
- **Accuracy & Evaluation** — gold-set precision/recall/F1, rule-vs-LLM comparison, per-domain accuracy and error analysis.

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm     # for in-text PII / NER

# Local LLM (one-time):
#   install Ollama, then:  ollama pull qwen2.5:3b
```

Python 3.12. `sqlite3`, `difflib`, `hashlib`, `json`, `re` are from the standard library.
For the cloud backend instead of Ollama: `pip install anthropic` + set `ANTHROPIC_API_KEY`.

> Prerequisite: Ollama must be running (`ollama serve` or the app open).

## Usage

```bash
# 1) Pipeline — clean, structure & load
python pipeline.py                 # full corpus with the LLM
python pipeline.py --limit 600     # domain-balanced sample (quick test)
python pipeline.py --rule          # offline rule-based extraction

# 2) Evaluation
python compare_llm.py --n 80       # rule-based vs LLM (coverage + agreement)
python make_gold.py                # build the gold test set
python evaluate.py                 # accuracy / precision / recall / F1 vs gold

# 3) Security scan
python scan_pii.py                 # detect & summarize in-text PII

# 4) Dashboard & assistant
streamlit run dashboard.py         # http://localhost:8501
python advisor.py "water pump leaking, low pressure"
```

---

## Data schemas

**Common schema (after `ingest`)**

```python
{record_id, domain, problem_raw, action_raw, date_raw, person_raw}
```

**Clean schema (pipeline output, table `maintenance`)**

```python
{record_id, domain, asset, failure_mode, action_type,
 date, person_id, problem_clean, quality}
```

> The aviation set arrives de-identified (no dates or person field); automotive/facility
> include dates. The pipeline handles this heterogeneity gracefully.

---

## Notes & limitations

- **Asset over-specification** — the small model finds the right component but names it
  too finely (e.g. `2-4-rocker-cover-gasket` vs `gasket`): asset exact-match is only 25%,
  though relaxed F1 is 0.73. Entity resolution is the planned fix.
- On the small categorical fields (failure / action) the rule-based extractor is
  competitive, which is why it is kept as a **hybrid fallback** rather than discarded.
- The gold set is modest (60 records); enlarging it would tighten the confidence intervals.
