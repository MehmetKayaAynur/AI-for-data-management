"""
Maintenance Data Dashboard
==========================
Raw MaintNet maintenance logs -> cleaned & structured database via a local LLM.
A worker-friendly dashboard:
  - Overview (KPIs + charts, all data)
  - Maintenance Records (worker note + extracted clean info, filtered here)
  - Before / After (raw messy data vs clean structured data)

Run:
    streamlit run dashboard.py
"""

import os
import re
import html
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

import pipeline as P   # raw data (ingest) + dictionaries
import advisor         # problem -> recommendation from past records

DB_PATH = "output/clean_maintenance.db"

st.set_page_config(page_title="Maintenance Dashboard", page_icon="🔧", layout="wide")

DOMAIN_COLORS = {"aviation": "#2563eb", "automotive": "#f59e0b", "facility": "#10b981"}
DOMAIN_ICON = {"aviation": "✈️", "automotive": "🚗", "facility": "🏭"}
DOMAIN_EN = {"aviation": "Aviation", "automotive": "Automotive", "facility": "Facility"}

C_ASSET, C_FAIL, C_ACT = "#2563eb", "#ef4444", "#10b981"


# ---------------------------------------------------------------------------
# Load data: clean (DB) + raw (source CSVs) merged on record_id
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    clean = pd.read_sql("SELECT * FROM maintenance", conn)
    conn.close()

    raw = P.ingest()[["record_id", "domain", "problem_raw", "action_raw", "date_raw"]]
    for d in (clean, raw):
        d["record_id"] = d["record_id"].astype(str)
        d["domain"] = d["domain"].astype(str)

    df = clean.merge(raw, on=["record_id", "domain"], how="left")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["asset_group"] = df["asset"].apply(_head_noun)   # core noun for charts
    return df


def _head_noun(asset):
    """'rocker-cover-gasket' -> 'gasket' (for chart distributions)."""
    if not isinstance(asset, str):
        return None
    toks = [t for t in re.findall(r"[a-z]+", asset.lower()) if len(t) > 2]
    return toks[-1] if toks else None


_ABBR_KEYS = set(P.ABBREV.keys())

def _has_abbr(text):
    toks = re.findall(r"[a-z/]+", str(text).lower())
    return any(t in _ABBR_KEYS for t in toks)


def pill(text, color):
    if not isinstance(text, str) or not text:
        return ""
    return (f'<span style="background:{color}22;color:{color};padding:2px 10px;'
            f'border-radius:12px;font-size:0.82em;font-weight:600;margin-right:6px;'
            f'white-space:nowrap">{html.escape(text)}</span>')


def dept_label(d):
    return f"{DOMAIN_ICON.get(d,'')} {DOMAIN_EN.get(d, d)}"


df = load_data()

# ---------------------------------------------------------------------------
if df is None or df.empty:
    st.title("🔧 Maintenance Dashboard")
    st.warning("Database not found. Run the pipeline first:\n\n"
               "`python pipeline.py`  (full data)  or  `python pipeline.py --limit 600`  (sample)")
    st.stop()

# ---------------------------------------------------------------------------
# Header + KPIs (whole dataset — no global filter)
# ---------------------------------------------------------------------------
st.title("🔧 Maintenance Logs")
st.caption("Messy maintenance notes written by workers → cleaned, structured and "
           "queryable data, powered by a local AI model")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Records", f"{len(df):,}")
k2.metric("Departments", df["domain"].nunique())
k3.metric("Unique assets", df["asset_group"].nunique())
k4.metric("Avg. quality", f"{df['quality'].mean():.2f}")
k5.metric("Records with date", f"{100*df['date'].notna().mean():.0f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📋 Maintenance Records", "🔄 Before / After",
     "🛠️ Assistant", "📈 Accuracy & Evaluation"])

# ===========================================================================
# TAB 1 — OVERVIEW (all data)
# ===========================================================================
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Department distribution")
        dom = df["domain"].value_counts().reset_index()
        dom.columns = ["domain", "count"]
        dom["dept"] = dom["domain"].map(dept_label)
        fig = px.pie(dom, names="dept", values="count", hole=0.45,
                     color="domain", color_discrete_map=DOMAIN_COLORS)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Quality score distribution")
        fig = px.histogram(df, x="quality", nbins=20, color="domain",
                           color_discrete_map=DOMAIN_COLORS)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320,
                          xaxis_title="quality score", yaxis_title="records",
                          bargap=0.05, legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Top 15 assets")
        top = df["asset_group"].value_counts().head(15).reset_index()
        top.columns = ["asset", "count"]
        fig = px.bar(top.sort_values("count"), x="count", y="asset", orientation="h")
        fig.update_traces(marker_color=C_ASSET)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=420,
                          xaxis_title="records", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.subheader("Most common failure modes")
        fm = df["failure_mode"].value_counts().head(15).reset_index()
        fm.columns = ["failure_mode", "count"]
        fig = px.bar(fm.sort_values("count"), x="count", y="failure_mode", orientation="h")
        fig.update_traces(marker_color=C_FAIL)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=420,
                          xaxis_title="records", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Most common actions")
        at = df["action_type"].value_counts().head(15).reset_index()
        at.columns = ["action_type", "count"]
        fig = px.bar(at.sort_values("count"), x="count", y="action_type", orientation="h")
        fig.update_traces(marker_color=C_ACT)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=420,
                          xaxis_title="records", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        st.subheader("Maintenance over time (monthly)")
        dated = df.dropna(subset=["date"])
        if dated.empty:
            st.info("No dated records (the aviation set has no date field).")
        else:
            ts = (dated.set_index("date").groupby("domain")
                  .resample("ME").size().reset_index(name="count"))
            ts["dept"] = ts["domain"].map(DOMAIN_EN)
            fig = px.line(ts, x="date", y="count", color="domain", markers=True,
                          color_discrete_map=DOMAIN_COLORS)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=420,
                              xaxis_title="", yaxis_title="records", legend_title="")
            st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 2 — MAINTENANCE RECORDS (filtered here, worker card view)
# ===========================================================================
with tab2:
    st.subheader("Maintenance records")
    st.caption("Each card: the worker's original note + the structured info extracted by AI.")

    fc = st.columns(4)
    depts = sorted(df["domain"].unique())
    sd = fc[0].selectbox("Department", ["(all)"] + depts,
                         format_func=lambda d: d if d == "(all)" else dept_label(d))
    view = df if sd == "(all)" else df[df["domain"] == sd]

    a_opts = ["(all)"] + view["asset_group"].value_counts().index.dropna().tolist()
    f_opts = ["(all)"] + view["failure_mode"].value_counts().index.dropna().tolist()
    c_opts = ["(all)"] + view["action_type"].value_counts().index.dropna().tolist()
    sa = fc[1].selectbox("Asset", a_opts)
    sf = fc[2].selectbox("Failure mode", f_opts)
    sc = fc[3].selectbox("Action", c_opts)

    fc2 = st.columns([3, 1])
    search = fc2[0].text_input("Search in notes (e.g. leak, pump)").strip().lower()
    show_n = fc2[1].number_input("How many", 5, 100, 20, step=5)

    if sa != "(all)":
        view = view[view["asset_group"] == sa]
    if sf != "(all)":
        view = view[view["failure_mode"] == sf]
    if sc != "(all)":
        view = view[view["action_type"] == sc]
    if search:
        view = view[view["problem_clean"].str.lower().str.contains(search, na=False)
                    | view["problem_raw"].str.lower().str.contains(search, na=False)]

    st.write(f"**{len(view):,}** records found — showing first {min(int(show_n), len(view))}.")

    for _, r in view.head(int(show_n)).iterrows():
        icon = DOMAIN_ICON.get(r["domain"], "")
        dept = DOMAIN_EN.get(r["domain"], r["domain"])
        date_txt = r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "no date"
        note = " · ".join([str(x) for x in (r.get("problem_raw"), r.get("action_raw"))
                           if isinstance(x, str) and x.strip()])
        pills = (pill(r.get("asset"), C_ASSET)
                 + pill(r.get("failure_mode"), C_FAIL)
                 + pill(r.get("action_type"), C_ACT))
        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"color:#6b7280;font-size:0.85em'><span>{icon} {dept} · #"
                f"{html.escape(str(r['record_id']))}</span><span>{date_txt} · "
                f"quality {r['quality']:.2f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin:6px 0'>{pills or '<i>no extraction</i>'}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"📝 <span style='color:#6b7280'>Worker's note:</span> "
                        f"{html.escape(note) if note else '—'}", unsafe_allow_html=True)

# ===========================================================================
# TAB 3 — BEFORE / AFTER (raw messy vs clean structured)
# ===========================================================================
with tab3:
    st.subheader("Before (raw) → After (clean, structured)")
    st.caption("The state of the same records before and after AI cleaning.")

    before_abbr = df["problem_raw"].apply(_has_abbr).mean()
    after_abbr = df["problem_clean"].apply(_has_abbr).mean()
    _shape = lambda s: re.sub(r"\d+", "N", str(s).strip())   # "1/25/19 0:00" -> "N/N/N N:N"
    before_fmt = max(1, df.loc[df["date_raw"].str.strip().ne(""), "date_raw"]
                        .apply(_shape).nunique())
    after_struct = df[["asset", "failure_mode", "action_type"]].notna().any(axis=1).mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records with abbreviations / jargon", f"{after_abbr*100:.0f}%",
              f"{(after_abbr-before_abbr)*100:.0f}% (was {before_abbr*100:.0f}%)",
              delta_color="inverse")
    m2.metric("Date format variety", "1 (ISO 8601)",
              f"was {before_fmt} different formats", delta_color="off")
    m3.metric("Records with structured fields", f"{after_struct*100:.0f}%",
              "was 0% (free text)")
    m4.metric("Data type", "SQL table", "was free text", delta_color="off")

    st.divider()
    st.markdown("#### Example transformations")
    st.caption("Left: the raw note written by the worker. Right: the structured record produced by AI. "
               "5 examples per department.")

    # 5 per department (aviation / automotive / facility)
    parts = []
    for d in sorted(df["domain"].unique()):
        sub = df[(df["domain"] == d) & df["asset"].notna()
                 & df["problem_raw"].apply(_has_abbr)]
        if len(sub) < 5:
            sub = df[(df["domain"] == d) & df["asset"].notna()]
        parts.append(sub.head(5))
    sample = pd.concat(parts) if parts else df.head(0)

    rows = []
    for _, r in sample.iterrows():
        rows.append({
            "Department": dept_label(r["domain"]),
            "🔴 Raw note (before)": " · ".join(
                [str(x) for x in (r.get("problem_raw"), r.get("action_raw"))
                 if isinstance(x, str) and x.strip()]),
            "🔴 Raw date": r.get("date_raw") or "—",
            "🟢 Asset": r.get("asset") or "—",
            "🟢 Failure": r.get("failure_mode") or "—",
            "🟢 Action": r.get("action_type") or "—",
            "🟢 Date": r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Abbreviation expansion examples")
    st.caption("Loaded automatically from MaintNet's own abbreviation dictionaries.")
    ex = list(P.ABBREV.items())[:12]
    st.dataframe(pd.DataFrame(ex, columns=["Raw (abbreviation)", "Expanded"]),
                 use_container_width=True, hide_index=True)

    # ---- Securing: in-text PII masking ----
    st.divider()
    st.markdown("#### Securing — in-text PII masking")
    st.caption("Real personal data hides inside the notes themselves. The pipeline detects "
               "person names (NER), phone numbers and e-mails (regex) and masks them. "
               "Raw PII values are never stored.")
    if os.path.exists("output/pii_findings.csv"):
        pii_df = pd.read_csv("output/pii_findings.csv", dtype=str).fillna("")
        n_person = pii_df["types"].str.contains("PERSON").sum()
        n_phone = pii_df["types"].str.contains("PHONE").sum()
        n_email = pii_df["types"].str.contains("EMAIL").sum()
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Records with PII", len(pii_df))
        q2.metric("Notes with a name", int(n_person))
        q3.metric("Notes with a phone", int(n_phone))
        q4.metric("Notes with an e-mail", int(n_email))
        st.caption("All findings fall in the **facility** domain — the aviation set is "
                   "de-identified, so the detector correctly flags nothing there.")
        st.dataframe(pii_df[["domain", "types", "masked_note"]].rename(
            columns={"domain": "Department", "types": "PII found", "masked_note": "Masked note"}),
            use_container_width=True, hide_index=True, height=300)
    else:
        st.info("Run `python scan_pii.py` to generate the PII report "
                "(`output/pii_findings.csv`).")

# ===========================================================================
# TAB 4 — ASSISTANT (problem -> recommendation from past records)
# ===========================================================================
with tab4:
    st.subheader("Describe a problem — get a recommendation")
    st.caption("Type a maintenance problem in plain text. The AI understands it and "
               "suggests what to do, based on similar past cases in the database.")

    q = st.text_area("Problem",
                     placeholder="e.g. left engine cylinder baffle cracked, oil leaking",
                     height=80)
    go = st.button("Get recommendation", type="primary")

    if go and q.strip():
        try:
            with st.spinner("Analyzing the problem and searching past records..."):
                res = advisor.advise(df, q.strip())
        except Exception as e:
            st.error(f"AI model not reachable ({e}). Is Ollama running? Try `ollama serve`.")
        else:
            f = res["fields"]
            st.markdown("**Understood as:** "
                        + (pill(f.get("asset"), C_ASSET) + pill(f.get("failure_mode"), C_FAIL)
                           or "—"), unsafe_allow_html=True)

            if res["recommended_action"]:
                dist = res["action_dist"]
                tot, n = sum(dist.values()), dist.get(res["recommended_action"], 0)
                st.success(f"**Recommended action: {res['recommended_action']}**  "
                           f"— used in {n}/{tot} similar past cases")

            if res["advice"]:
                st.markdown("**AI advice (based on past cases):**")
                st.info(res["advice"])

            m = res["matches"]
            if not m.empty:
                st.markdown(f"#### Evidence — {len(m)} similar past records")
                ev = m[["domain", "asset", "failure_mode", "action_type",
                        "date", "problem_clean"]].copy()
                ev["domain"] = ev["domain"].map(DOMAIN_EN)
                ev["date"] = ev["date"].dt.strftime("%Y-%m-%d").fillna("—")
                st.dataframe(ev, use_container_width=True, hide_index=True)
            else:
                st.warning("No similar past records found for this problem.")
    elif go:
        st.warning("Please type a problem first.")

# ===========================================================================
# TAB 5 — ACCURACY & EVALUATION (gold-set metrics, comparison, criteria)
# ===========================================================================
with tab5:
    st.subheader("Accuracy & evaluation")
    st.caption("How *correct* is the AI — not just how often it fills a field. Measured "
               "against a hand-labeled gold set of 60 records (20 per domain), with a "
               "rule-based vs. LLM comparison.")

    try:
        import evaluate as E
        res, perrec = E.compute()
    except Exception as e:
        st.warning("Gold set not available yet. Run `python make_gold.py` then "
                   f"`python evaluate.py`.\n\n({e})")
        res = None

    if res is not None:
        FNAME = {"asset": "Asset", "failure_mode": "Failure mode", "action_type": "Action type"}
        macro = res.groupby("method")[["f1", "acc_exact"]].mean()
        llm_f1, rule_f1 = macro.loc["LLM", "f1"], macro.loc["RULE", "f1"]
        asset_llm = res[(res.field == "asset") & (res.method == "LLM")]["f1"].iloc[0]

        # ---- numeric KPIs ----
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Gold records", len(perrec))
        e2.metric("LLM macro-F1", f"{llm_f1*100:.0f}%",
                  f"{(llm_f1-rule_f1)*100:+.0f} pts vs rule-based")
        e3.metric("Rule-based macro-F1", f"{rule_f1*100:.0f}%")
        e4.metric("Asset F1 (LLM)", f"{asset_llm:.2f}", "vs 0.33 rule-based")

        st.divider()

        # ---- comparison chart: F1 per field, rule vs LLM ----
        st.markdown("#### Rule-based vs. LLM — F1 per field (relaxed match)")
        cd = res.copy()
        cd["F1"] = (cd["f1"] * 100).round(0)
        cd["Field"] = cd["field"].map(FNAME)
        cd["Method"] = cd["method"].map({"RULE": "Rule-based", "LLM": "LLM (qwen2.5:3b)"})
        fig = px.bar(cd, x="Field", y="F1", color="Method", barmode="group", text="F1",
                     color_discrete_map={"Rule-based": "#2563eb",
                                         "LLM (qwen2.5:3b)": "#f59e0b"})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=360, yaxis_title="F1 (%)", xaxis_title="",
                          yaxis_range=[0, 109], legend_title="",
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("LLM wins decisively on **asset**; on the small categorical fields "
                   "(failure / action) the rule-based method stays competitive — which is "
                   "why the pipeline keeps it as a fallback (hybrid design).")

        # ---- full metrics table ----
        st.markdown("#### Full metrics")
        show = res.copy()
        show["field"] = show["field"].map(FNAME)
        show["method"] = show["method"].map({"RULE": "Rule-based", "LLM": "LLM"})
        for c in ["coverage", "acc_exact", "acc_relaxed", "precision", "recall", "f1"]:
            show[c] = (show[c] * 100).round(0).astype(int).astype(str) + "%"
        show = show.rename(columns={"field": "Field", "method": "Method",
                                    "coverage": "Coverage", "acc_exact": "Exact",
                                    "acc_relaxed": "Relaxed", "precision": "Precision",
                                    "recall": "Recall", "f1": "F1"})
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("**Exact** = identical string · **Relaxed** = same head noun / shared word "
                   "(credits e.g. `gasket` ~ `intake-gasket`). Precision/Recall/F1 use relaxed match.")

        # ---- accuracy by domain ----
        st.divider()
        st.markdown("#### Accuracy by domain (LLM, relaxed match)")
        pdom = E.per_domain(perrec, "llm")
        melt = pdom.melt(id_vars="domain", value_vars=["asset", "failure_mode", "action_type"],
                         var_name="field", value_name="acc")
        melt["acc"] = (melt["acc"] * 100).round(0)
        melt["Department"] = melt["domain"].map(DOMAIN_EN).fillna(melt["domain"])
        cmap = {DOMAIN_EN.get(k, k): v for k, v in DOMAIN_COLORS.items()}
        figd = px.bar(melt, x="field", y="acc", color="Department", barmode="group",
                      text="acc", color_discrete_map=cmap)
        figd.update_traces(textposition="outside")
        figd.update_layout(height=340, yaxis_title="accuracy (%)", xaxis_title="",
                           yaxis_range=[0, 109], legend_title="",
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(figd, use_container_width=True)
        best = pdom.loc[pdom["overall"].idxmax(), "domain"]
        worst = pdom.loc[pdom["overall"].idxmin(), "domain"]
        st.caption(f"Accuracy varies by domain — **{DOMAIN_EN.get(worst, worst)}** notes are the "
                   f"hardest (often very terse, e.g. “DONT START”), while **{DOMAIN_EN.get(best, best)}** "
                   "scores highest.")

        # ---- error analysis ----
        st.markdown("#### Error analysis — where the LLM disagrees with the gold labels")
        mm = E.mismatches(perrec, "llm")
        st.write(f"**{len(mm)}** of {len(perrec)} gold records differ on at least one field — "
                 "most are asset over-specification (the right part, named too finely).")
        st.dataframe(mm, use_container_width=True, hide_index=True, height=320)


# ---------------------------------------------------------------------------
# Download (full clean dataset)
# ---------------------------------------------------------------------------
st.divider()
dl_cols = ["record_id", "domain", "asset", "failure_mode", "action_type",
           "date", "person_id", "quality", "problem_clean"]
st.download_button("Download clean dataset as CSV",
                   df[dl_cols].to_csv(index=False).encode("utf-8"),
                   "clean_maintenance.csv", "text/csv")
