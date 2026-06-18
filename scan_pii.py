"""Scan the whole corpus for in-text PII and write a summary.

For every record the worker's note (problem + action) is scanned for person names,
phone numbers and e-mails; the note is stored MASKED (raw PII values are never
persisted). Produces output/pii_findings.csv + console summary.

Usage: python scan_pii.py
"""
import pandas as pd
import pipeline as P
import pii


def main():
    df = P.ingest()
    notes = [" · ".join(x for x in (p, a) if isinstance(x, str) and x.strip())
             for p, a in zip(df["problem_raw"], df["action_raw"])]
    print(f"Scanning {len(notes)} notes for PII (NER: "
          f"{'spaCy' if pii._nlp() is not None else 'regex fallback'}) ...")
    findings = pii.detect_batch(notes)

    rows, n_person, n_phone, n_email = [], 0, 0, 0
    for rid, dom, note, finds in zip(df["record_id"], df["domain"], notes, findings):
        if not finds:
            continue
        types = [t for t, _ in finds]
        n_person += types.count("PERSON")
        n_phone += types.count("PHONE")
        n_email += types.count("EMAIL")
        masked = note
        for typ, val in finds:
            import re
            masked = re.sub(re.escape(val), f"[{typ}]", masked, flags=re.IGNORECASE)
        rows.append({"record_id": rid, "domain": dom,
                     "n_pii": len(finds),
                     "types": ", ".join(sorted(set(types))),
                     "masked_note": masked[:160]})
    out = pd.DataFrame(rows)
    out.to_csv("output/pii_findings.csv", index=False)

    total = len(out)
    print("\n=== PII SCAN SUMMARY ===")
    print(f"  Records containing PII : {total}  (of {len(df)}, %{100*total/len(df):.1f})")
    print(f"  Person names found     : {n_person}")
    print(f"  Phone numbers found    : {n_phone}")
    print(f"  E-mails found          : {n_email}")
    if total:
        print("\n  By domain:")
        print(out["domain"].value_counts().to_string())
        print("\n  Example masked notes:")
        for _, r in out.head(6).iterrows():
            print(f"   [{r['domain']}] {r['masked_note']}")
    print("\nOutput: output/pii_findings.csv  (raw PII values are NOT stored)")


if __name__ == "__main__":
    main()
