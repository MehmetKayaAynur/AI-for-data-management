"""
Kural-tabanli vs LLM cikarim kiyaslamasi
========================================
Ayni temizlenmis bakim kayitlari uzerinde iki yontemi calistirir:
  1) rule_based_extract  (sozluk tabanli, offline)
  2) llm_extract         (Anthropic API, baglamsal)

Olcer:
  - Coverage : her yontemin alan basina doldurma orani (% non-null)
  - Agreement: iki yontemin AYNI degeri urettigi kayit orani
              (yalniz her ikisinin de doldurdugu kayitlar uzerinden)

NOT: Burada "dogruluk" (accuracy) degil, "kapsama" + "uyum" olculur.
     Gercek accuracy icin elle etiketli altin (gold) kume gerekir; bunun
     icin compare_llm.py --make-gold ile sablon uretebilirsin.

Kullanim:
  python compare_llm.py --n 80           # 80 kayitlik ornek uzerinde kiyasla
  python compare_llm.py --n 80 --domain aviation
  python compare_llm.py --make-gold 40   # elle etiketleme sablonu uret

Onkosul: pip install anthropic  +  ortamda ANTHROPIC_API_KEY tanimli olmali.
"""

import os
import re
import sys
import time
import argparse

import pandas as pd

import pipeline as P

RESULTS_CSV = "output/llm_vs_rule.csv"
CHART_PNG = "output/llm_vs_rule.png"
GOLD_TEMPLATE = "output/gold_template.csv"

FIELDS = ["asset", "failure_mode", "action_type"]


# ---------------------------------------------------------------------------
# Ortak: temizlenmis ornek kayitlari hazirla
# ---------------------------------------------------------------------------
def prepare_sample(n, domain=None, seed=42):
    df = P.ingest()
    if domain:
        df = df[df["domain"] == domain].reset_index(drop=True)
    if n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "record_id": row["record_id"],
            "domain": row["domain"],
            "problem_clean": P.expand_and_correct(row["problem_raw"]),
            "action_clean": P.expand_and_correct(row["action_raw"]),
        })
    return rows


# ---------------------------------------------------------------------------
# Elle etiketleme sablonu (gercek accuracy olcmek isteyenler icin)
# ---------------------------------------------------------------------------
def make_gold(n, domain=None):
    os.makedirs("output", exist_ok=True)
    rows = prepare_sample(n, domain)
    out = pd.DataFrame([{
        "record_id": r["record_id"], "domain": r["domain"],
        "problem_clean": r["problem_clean"], "action_clean": r["action_clean"],
        "gold_asset": "", "gold_failure_mode": "", "gold_action_type": "",
    } for r in rows])
    out.to_csv(GOLD_TEMPLATE, index=False)
    print(f"Elle etiketleme sablonu yazildi: {GOLD_TEMPLATE}")
    print("gold_* kolonlarini doldurup gercek accuracy hesabinda kullanabilirsin.")


# ---------------------------------------------------------------------------
# Kiyas
# ---------------------------------------------------------------------------
def compare(n, domain=None):
    rows = prepare_sample(n, domain)
    print(f"Ornek: {len(rows)} kayit"
          + (f" (domain={domain})" if domain else "") + f"  |  LLM modeli: {P.LLM_MODEL}")

    # LLM erisilebilir mi? (backend'e gore)
    try:
        P._get_llm_client()
    except ImportError:
        pkg = "ollama" if P.LLM_BACKEND == "ollama" else "anthropic"
        sys.exit(f"HATA: '{pkg}' paketi yok. Kurulum: pip install {pkg}")
    except Exception as e:
        if P.LLM_BACKEND == "ollama":
            sys.exit(f"HATA: Ollama istemcisi olusturulamadi ({e}). "
                     "Ollama kurulu ve calisiyor mu? (ollama serve)")
        sys.exit(f"HATA: Anthropic istemcisi olusturulamadi ({e}). "
                 "ANTHROPIC_API_KEY tanimli mi?")

    records = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        rule = P.rule_based_extract(r["problem_clean"], r["action_clean"])
        try:
            llm = P.llm_extract(r["problem_clean"], r["action_clean"])
        except Exception as e:
            print(f"  [{i}/{len(rows)}] LLM cagrisi basarisiz: {e}")
            llm = {f: None for f in FIELDS}
        rec = {"record_id": r["record_id"], "domain": r["domain"],
               "problem_clean": r["problem_clean"]}
        for f in FIELDS:
            rec[f"rule_{f}"] = rule[f]
            rec[f"llm_{f}"] = llm[f]
        records.append(rec)
        if i % 10 == 0:
            print(f"  {i}/{len(rows)} islendi...")

    elapsed = time.time() - t0
    out = pd.DataFrame(records)
    os.makedirs("output", exist_ok=True)
    out.to_csv(RESULTS_CSV, index=False)

    # --- metrikler ---
    print(f"\n=== KIYAS SONUCU ({len(out)} kayit, {elapsed:.1f}s) ===\n")
    cov_rule, cov_llm, agree_exact, agree_loose = {}, {}, {}, {}
    for f in FIELDS:
        rcol, lcol = out[f"rule_{f}"], out[f"llm_{f}"]
        cov_rule[f] = rcol.notna().mean()
        cov_llm[f] = lcol.notna().mean()
        both = rcol.notna() & lcol.notna()
        if both.sum():
            pairs = list(zip(rcol[both], lcol[both]))
            agree_exact[f] = sum(_exact(a, b) for a, b in pairs) / len(pairs)
            agree_loose[f] = sum(_loose(a, b) for a, b in pairs) / len(pairs)
        else:
            agree_exact[f] = agree_loose[f] = float("nan")

    hdr = (f"{'Alan':<14}{'Kural kaps.':>13}{'LLM kaps.':>11}"
           f"{'Uyum(tam)':>12}{'Uyum(gevsek)':>14}")
    print(hdr)
    print("-" * len(hdr))
    for f in FIELDS:
        print(f"{f:<14}{cov_rule[f]*100:>11.0f}% {cov_llm[f]*100:>9.0f}% "
              f"{agree_exact[f]*100:>10.0f}% {agree_loose[f]*100:>12.0f}%")

    _chart(cov_rule, cov_llm, agree_exact, agree_loose)
    print(f"\nCikti: {RESULTS_CSV}  ve  {CHART_PNG}")
    print("Not: Uyum(tam)    = iki yontem AYNEN ayni degeri uretti.")
    print("     Uyum(gevsek) = en az bir ortak kelime (or. 'gasket' ~ 'intake-gasket').")


def _exact(a, b):
    return str(a).strip().lower() == str(b).strip().lower()

def _loose(a, b):
    """Gevsek uyum: iki deger en az bir anlamli (>2 harf) kelimeyi paylasiyorsa."""
    ta = set(t for t in re.findall(r"[a-z]+", str(a).lower()) if len(t) > 2)
    tb = set(t for t in re.findall(r"[a-z]+", str(b).lower()) if len(t) > 2)
    return bool(ta & tb)


def _chart(cov_rule, cov_llm, agree_exact, agree_loose):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    series = [
        ("Kural kapsama", cov_rule),
        ("LLM kapsama", cov_llm),
        ("Uyum (tam)", agree_exact),
        ("Uyum (gevsek)", agree_loose),
    ]
    x = np.arange(len(FIELDS))
    w = 0.2
    offs = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, d), off in zip(series, offs):
        vals = [(d[f]*100 if d[f] == d[f] else 0) for f in FIELDS]
        bars = ax.bar(x + off, vals, w, label=label)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + w/2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_ylabel("%")
    ax.set_title(f"Kural-tabanli vs LLM cikarim  ({P.LLM_MODEL})")
    ax.set_xticks(x)
    ax.set_xticklabels(FIELDS)
    ax.set_ylim(0, 108)
    ax.legend(ncol=4, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(CHART_PNG, dpi=120, bbox_inches="tight")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="ornek kayit sayisi")
    ap.add_argument("--domain", choices=["aviation", "automotive", "facility"],
                    help="tek bir domaine kisitla")
    ap.add_argument("--make-gold", type=int, metavar="N",
                    help="N kayitlik elle etiketleme sablonu uret, kiyas yapma")
    args = ap.parse_args()

    if args.make_gold:
        make_gold(args.make_gold, args.domain)
    else:
        compare(args.n, args.domain)
