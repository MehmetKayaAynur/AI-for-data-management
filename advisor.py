"""
Bakim Danismani (Maintenance Advisor)
=====================================
Kullanici serbest metin bir SORUN girer; sistem:
  1) Sorunu yapilandirir (asset + failure_mode)            -- LLM
  2) Temiz veritabanindan benzer GECMIS kayitlari bulur    -- retrieval
  3) O kayitlarda yapilan islemlere bakarak ne yapilmasi
     gerektigini onerir                                    -- aggregation
  4) Gecmis vakalara dayanarak kisa, dogal bir tavsiye uretir (RAG) -- LLM

Bu, yapilandirilmis verinin DEGERINI gosterir: ham loglarla bu mumkun degildi.
"""

import re

import pandas as pd

import pipeline as P

ADVICE_SYS = (
    "You are a maintenance assistant for factory technicians. "
    "Given a new problem and similar past resolved cases from the maintenance "
    "database, recommend the next concrete maintenance action(s) as 1-3 short "
    "bullet points. Base your advice ONLY on the past cases provided. "
    "Be brief, practical, and specific."
)


def _head_noun(asset):
    if not isinstance(asset, str):
        return None
    toks = [t for t in re.findall(r"[a-z]+", asset.lower()) if len(t) > 2]
    return toks[-1] if toks else None


def _tokens(s):
    return set(t for t in re.findall(r"[a-z]+", str(s).lower()) if len(t) > 2)


def understand(problem_text):
    """Serbest metin sorunu yapilandirir -> (temiz_metin, {asset, failure_mode, action_type})."""
    clean = P.expand_and_correct(problem_text)
    fields = P.llm_extract(clean, "")     # sadece sorun var, aksiyon bos
    return clean, fields


def find_similar(df, problem_clean, asset, failure_mode, k=8):
    """Benzer gecmis kayitlari skorlayarak getirir (asset > failure > metin ortusmesi)."""
    d = df.copy()
    if "asset_group" not in d.columns:
        d["asset_group"] = d["asset"].apply(_head_noun)
    ag = _head_noun(asset)
    q_tokens = _tokens(problem_clean)

    score = pd.Series(0.0, index=d.index)
    if ag:
        score += (d["asset_group"] == ag).astype(float) * 3.0
    if failure_mode:
        score += (d["failure_mode"] == failure_mode).astype(float) * 2.0
    overlap = d["problem_clean"].apply(lambda t: len(q_tokens & _tokens(t)))
    score += overlap.clip(upper=3) * 0.5

    d = d.assign(_score=score)
    d = d[d["_score"] > 0].sort_values("_score", ascending=False)
    return d.head(k)


def recommend_action(matches):
    """Benzer kayitlardaki action_type dagilimindan en sik islemi secer."""
    acts = matches["action_type"].dropna()
    if acts.empty:
        return None, {}
    counts = acts.value_counts()
    return counts.index[0], counts.to_dict()


def llm_advice(problem_text, matches, max_cases=8):
    """Gecmis vakalara dayanarak LLM ile kisa tavsiye uretir (RAG)."""
    rows = []
    for r in matches.head(max_cases).itertuples():
        act = getattr(r, "action_type", None) or "?"
        rows.append(f"- Problem: {r.problem_clean} | Action taken: {act}")
    cases = "\n".join(rows) if rows else "(no similar cases found)"
    prompt = (f"New problem: {problem_text}\n\n"
              f"Similar past resolved cases:\n{cases}\n\n"
              "Recommend the next maintenance action(s):")

    client = P._get_llm_client()
    if P.LLM_BACKEND == "ollama":
        resp = client.chat(
            model=P.LLM_MODEL,
            messages=[{"role": "system", "content": ADVICE_SYS},
                      {"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        return resp["message"]["content"].strip()
    resp = client.messages.create(
        model=P.LLM_MODEL, max_tokens=300, system=ADVICE_SYS,
        messages=[{"role": "user", "content": prompt}])
    return next(b.text for b in resp.content if b.type == "text").strip()


def advise(df, problem_text, k=8):
    """Uctan uca: anla -> benzer bul -> oner -> tavsiye uret."""
    clean, fields = understand(problem_text)
    matches = find_similar(df, clean, fields.get("asset"), fields.get("failure_mode"), k)
    rec_action, dist = recommend_action(matches)
    advice = llm_advice(problem_text, matches) if not matches.empty else None
    return {
        "clean": clean,
        "fields": fields,
        "matches": matches,
        "recommended_action": rec_action,
        "action_dist": dist,
        "advice": advice,
    }


# CLI testi: python advisor.py "left engine cylinder baffle cracked, oil leaking"
if __name__ == "__main__":
    import sys
    import sqlite3

    q = " ".join(sys.argv[1:]) or "left engine cylinder baffle cracked, oil leaking"
    conn = sqlite3.connect(P.DB_PATH)
    df = pd.read_sql("SELECT * FROM maintenance", conn)
    conn.close()
    df["asset_group"] = df["asset"].apply(_head_noun)

    print(f'SORUN: "{q}"\n')
    res = advise(df, q)
    print("Anlasildi  :", res["fields"])
    print("Onerilen islem:", res["recommended_action"], res["action_dist"])
    print("\nAI tavsiyesi:\n", res["advice"])
    print(f"\n{len(res['matches'])} benzer gecmis kayit:")
    print(res["matches"][["domain", "asset", "failure_mode", "action_type", "problem_clean"]]
          .head(6).to_string(index=False))
