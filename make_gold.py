"""Build a hand-labeled GOLD test set for accuracy evaluation.

60 records (20 per domain), labeled by reading the raw note. Each record gets a
canonical gold value for asset / failure_mode / action_type (or 'none').
Produces output/gold_labeled.csv used by evaluate.py.
"""
import pandas as pd
import pipeline as P

# record_id -> (gold_asset, gold_failure_mode, gold_action_type)   ('none' = absent)
GOLD = {
    # ---- automotive ----
    "14347": ("bolt", "missing", "none"),
    "14520": ("none", "no_charge", "none"),
    "14335": ("mirror", "missing", "none"),
    "14292": ("none", "broken", "none"),
    "14310": ("tire", "none", "inspect_clean"),
    "14006": ("pin", "missing", "none"),
    "14653": ("sensor", "no_function", "replace"),
    "14567": ("panel", "loose", "none"),
    "14474": ("coolant", "low_level", "inspect_clean"),
    "14430": ("light", "broken", "none"),
    "14686": ("wiper", "none", "inspect_clean"),
    "14574": ("strobe", "no_function", "none"),
    "14905": ("none", "no_function", "none"),
    "14466": ("starter", "none", "inspect_clean"),
    "14694": ("wheel", "noise", "inspect_clean"),
    "14762": ("light", "broken", "none"),
    "14378": ("light", "broken", "none"),
    "14589": ("brake", "no_function", "none"),
    "14529": ("oil", "low_level", "none"),
    "14368": ("plow", "no_function", "none"),
    # ---- aviation ----
    "102745": ("baffle", "crack", "repair"),
    "103804": ("gasket", "leak", "replace"),
    "103224": ("gasket", "leak", "replace"),
    "103204": ("gasket", "leak", "inspect_clean"),
    "103280": ("gasket", "leak", "replace"),
    "105069": ("bolt", "loose", "secure"),
    "100399": ("gasket", "leak", "replace"),
    "103379": ("gasket", "leak", "replace"),
    "104961": ("engine", "none", "inspect_clean"),
    "103482": ("standoff", "broken", "replace"),
    "103642": ("gasket", "leak", "replace"),
    "102927": ("seal", "loose", "secure"),
    "105805": ("prop", "none", "inspect_clean"),
    "102497": ("engine", "damage", "inspect_clean"),
    "105300": ("gasket", "leak", "replace"),
    "104538": ("gasket", "leak", "replace"),
    "100988": ("gasket", "leak", "replace"),
    "100122": ("gasket", "leak", "replace"),
    "100522": ("baffle", "crack", "fabricate"),
    "105354": ("plug", "damage", "replace"),
    # ---- facility ----
    "113794": ("tile", "damage", "replace"),
    "113514": ("camera", "no_function", "none"),
    "114343": ("none", "none", "none"),
    "114370": ("drywall", "damage", "repair"),
    "113283": ("door", "none", "align"),
    "114395": ("none", "no_function", "none"),
    "113510": ("window", "none", "repair"),
    "113769": ("door", "none", "repair"),
    "113990": ("generator", "none", "inspect_clean"),
    "114012": ("boiler", "none", "inspect_clean"),
    "113589": ("tile", "damage", "replace"),
    "113757": ("none", "none", "none"),
    "113458": ("boiler", "no_function", "none"),
    "114001": ("none", "none", "inspect_clean"),
    "113310": ("lock", "none", "none"),
    "113419": ("paddle", "no_function", "none"),
    "114327": ("tile", "damage", "replace"),
    "113722": ("none", "none", "none"),
    "113423": ("heater", "none", "repair"),
    "114132": ("none", "none", "none"),
}


def main():
    df = P.ingest()
    df["record_id"] = df["record_id"].astype(str)
    sub = df[df["record_id"].isin(GOLD)].copy()
    rows = []
    for _, r in sub.iterrows():
        rid = r["record_id"]
        g = GOLD[rid]
        rows.append({
            "record_id": rid,
            "domain": r["domain"],
            "problem_clean": P.expand_and_correct(r["problem_raw"]),
            "action_clean": P.expand_and_correct(r["action_raw"]),
            "gold_asset": g[0],
            "gold_failure_mode": g[1],
            "gold_action_type": g[2],
        })
    out = pd.DataFrame(rows)
    out.to_csv("output/gold_labeled.csv", index=False)
    print(f"Wrote output/gold_labeled.csv with {len(out)} labeled records "
          f"({out['domain'].value_counts().to_dict()})")
    missing = set(GOLD) - set(out["record_id"])
    if missing:
        print("WARNING: ids not found in data:", missing)


if __name__ == "__main__":
    main()
