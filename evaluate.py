"""Accuracy evaluation against a hand-labeled GOLD set.

Compares the rule-based extractor and the LLM extractor to ground truth and
reports, per field (asset / failure_mode / action_type):
  coverage, exact accuracy, relaxed accuracy, precision, recall, F1.

LLM predictions are read from the already-computed output/clean_maintenance.csv
(no need to re-run the model). Rule-based predictions are computed on the fly.

Usage:  python evaluate.py
Output: console table + output/accuracy_eval.csv + output/accuracy_eval.png
"""
import re
import pandas as pd
import pipeline as P

GOLD_CSV = "output/gold_labeled.csv"
CLEAN_CSV = "output/clean_maintenance.csv"
OUT_CSV = "output/accuracy_eval.csv"
OUT_PNG = "output/accuracy_eval.png"
FIELDS = ["asset", "failure_mode", "action_type"]


def norm(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    return None if s in ("", "none", "nan", "null", "n/a") else s


def toks(v):
    return set(t for t in re.findall(r"[a-z]+", str(v).lower()) if len(t) > 2)


def head(v):
    """Head noun: last meaningful token of a (possibly compound) value."""
    ts = [t for t in re.findall(r"[a-z]+", str(v).lower()) if len(t) > 2]
    return ts[-1] if ts else None


def exact(pred, gold):
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    return norm(pred) == norm(gold)


def relaxed(pred, gold):
    """Credit a match if head nouns agree or the values share a meaningful token."""
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False
    if head(pred) == head(gold):
        return True
    return bool(toks(pred) & toks(gold))


def prf(preds, golds, match):
    tp = fp = fn = 0
    for pr, gl in zip(preds, golds):
        if gl is not None and pr is not None and match(pr, gl):
            tp += 1
        elif pr is not None and (gl is None or not match(pr, gl)):
            fp += 1
        elif gl is not None and (pr is None or not match(pr, gl)):
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def compute(gold_csv=GOLD_CSV, clean_csv=CLEAN_CSV):
    """Returns (res, out): per-field/method metrics and per-record predictions.

    No files are written and no model is called (LLM predictions come from the
    already-computed clean CSV), so this is safe to import from the dashboard.
    """
    gold = pd.read_csv(gold_csv, dtype=str).fillna("")
    clean = pd.read_csv(clean_csv, dtype=str).fillna("")
    clean["record_id"] = clean["record_id"].astype(str)
    llm_by_id = clean.set_index("record_id")

    records = []
    for _, r in gold.iterrows():
        rid = str(r["record_id"])
        rule = P.rule_based_extract(r["problem_clean"], r["action_clean"])
        llm = {f: (llm_by_id.loc[rid, f] if rid in llm_by_id.index else None)
               for f in FIELDS}
        rec = {"record_id": rid, "domain": r["domain"],
               "problem_clean": r.get("problem_clean", "")}
        for f in FIELDS:
            rec[f"gold_{f}"] = norm(r[f"gold_{f}"])
            rec[f"rule_{f}"] = norm(rule[f])
            rec[f"llm_{f}"] = norm(llm[f])
        records.append(rec)
    out = pd.DataFrame(records)

    rows = []
    for f in FIELDS:
        g = out[f"gold_{f}"].tolist()
        for method in ("rule", "llm"):
            p = out[f"{method}_{f}"].tolist()
            cov = sum(x is not None for x in p) / len(p)
            acc_e = sum(exact(a, b) for a, b in zip(p, g)) / len(p)
            acc_r = sum(relaxed(a, b) for a, b in zip(p, g)) / len(p)
            pr, rc, f1 = prf(p, g, relaxed)
            rows.append({"field": f, "method": method.upper(),
                         "coverage": cov, "acc_exact": acc_e, "acc_relaxed": acc_r,
                         "precision": pr, "recall": rc, "f1": f1})
    res = pd.DataFrame(rows)
    return res, out


def per_domain(out, method="llm"):
    """Per-domain relaxed-match accuracy of one method, per field + overall."""
    rows = []
    for dom, g in out.groupby("domain"):
        rec = {"domain": dom, "n": len(g)}
        accs = []
        for f in FIELDS:
            a = sum(relaxed(p, gl) for p, gl in zip(g[f"{method}_{f}"], g[f"gold_{f}"])) / len(g)
            rec[f] = a
            accs.append(a)
        rec["overall"] = sum(accs) / len(accs)
        rows.append(rec)
    return pd.DataFrame(rows)


def mismatches(out, method="llm"):
    """Records where the method disagrees with gold on at least one field."""
    rows = []
    for _, r in out.iterrows():
        wrong = [f for f in FIELDS if not relaxed(r[f"{method}_{f}"], r[f"gold_{f}"])]
        if not wrong:
            continue
        rec = {"domain": r["domain"], "note": r.get("problem_clean", ""),
               "wrong fields": ", ".join(f.replace("_", " ") for f in wrong)}
        for f in FIELDS:
            rec[f"{f} (gold→pred)"] = f"{r[f'gold_{f}'] or '—'} → {r[f'{method}_{f}'] or '—'}"
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    res, out = compute()
    res.to_csv(OUT_CSV, index=False)

    # ---- print ----
    print(f"\nGOLD accuracy evaluation — {len(out)} hand-labeled records "
          f"(20 per domain)\n")
    hdr = (f"{'field':<13}{'method':<7}{'cover':>7}{'exact':>8}{'relax':>8}"
           f"{'prec':>7}{'rec':>7}{'F1':>7}")
    print(hdr); print("-" * len(hdr))
    for _, r in res.iterrows():
        print(f"{r['field']:<13}{r['method']:<7}{r['coverage']*100:>6.0f}%"
              f"{r['acc_exact']*100:>7.0f}%{r['acc_relaxed']*100:>7.0f}%"
              f"{r['precision']*100:>6.0f}%{r['recall']*100:>6.0f}%{r['f1']*100:>6.0f}%")

    # macro F1
    print()
    for method in ("RULE", "LLM"):
        sub = res[res["method"] == method]
        print(f"  macro-F1 ({method}, relaxed): {sub['f1'].mean()*100:.0f}%   "
              f"macro exact-acc: {sub['acc_exact'].mean()*100:.0f}%")

    _chart(res)
    print(f"\nOutput: {OUT_CSV}  and  {OUT_PNG}")


def _chart(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(FIELDS)); w = 0.36
    for ax, metric, title in [(ax1, "f1", "F1 score (relaxed match)"),
                              (ax2, "acc_exact", "Exact-match accuracy")]:
        rule = [res[(res.field == f) & (res.method == "RULE")][metric].iloc[0]*100 for f in FIELDS]
        llm = [res[(res.field == f) & (res.method == "LLM")][metric].iloc[0]*100 for f in FIELDS]
        b1 = ax.bar(x - w/2, rule, w, label="Rule-based", color="#2563eb")
        b2 = ax.bar(x + w/2, llm, w, label="LLM (qwen2.5:3b)", color="#f59e0b")
        for bars in (b1, b2):
            for r in bars:
                ax.text(r.get_x()+w/2, r.get_height()+1, f"{r.get_height():.0f}",
                        ha="center", fontsize=8)
        ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(FIELDS, fontsize=9)
        ax.set_ylim(0, 109); ax.set_ylabel("%")
    ax1.legend(loc="lower center", bbox_to_anchor=(1.1, -0.28), ncol=2, frameon=False)
    fig.suptitle("Accuracy vs. hand-labeled gold set (60 records)", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
