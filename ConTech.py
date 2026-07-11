import re
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title="Construction Startup Directory",
    page_icon="🏗️",
    layout="wide"
)

st.title("Construction Startup Directory Chatbot")
st.caption("Ask natural-language questions about construction tech startups.")

CSV_PATH = "data/startups.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

def normalize_columns(cols):
    return {str(c).lower().strip(): c for c in cols}

def get_col(df, candidates):
    lookup = normalize_columns(df.columns)
    for c in candidates:
        if c in lookup:
            return lookup[c]
    return None

def clean_text(x):
    return "" if pd.isna(x) else str(x)

def detect_intent(q):
    ql = q.lower().strip()
    if re.search(r"\b(how many|count|number of)\b", ql):
        return "count"
    if re.search(r"\b(compare|vs\.?|versus)\b", ql):
        return "compare"
    if re.search(r"\b(what funding stages|funding stages represented|which stages)\b", ql):
        return "stages"
    return "search"

def detect_location(prompt, location_values):
    q = prompt.lower()
    for loc in location_values:
        locs = str(loc).strip()
        if not locs:
            continue
        if locs.lower() in q:
            return locs
    return None

def detect_keywords(prompt, keyword_values):
    q = prompt.lower()
    hits = []
    for kw in keyword_values:
        kws = str(kw).strip()
        if not kws:
            continue
        if kws.lower() in q:
            hits.append(kws)
    return hits

def apply_text_filters(df, filters):
    out = df.copy()

    if filters.get("location") and filters.get("location_col"):
        loc = filters["location"]
        col = filters["location_col"]
        out = out[
            out[col].astype(str).str.contains(loc, case=False, na=False, regex=False)
        ]

    for kw in filters.get("domain_hits", []):
        mask = (
            out[filters["category_col"]].astype(str).str.contains(kw, case=False, na=False, regex=False)
            | out[filters["desc_col"]].astype(str).str.contains(kw, case=False, na=False, regex=False)
            | out["search_text"].astype(str).str.contains(kw, case=False, na=False, regex=False)
        )
        out = out[mask]

    return out

def build_embeddings_for_df(df):
    model = load_model()
    return model.encode(df["search_text"].tolist(), convert_to_tensor=True)

def top_matches(df, query, k=TOP_K):
    if df.empty:
        return df.copy()

    model = load_model()
    embeddings = build_embeddings_for_df(df)
    qvec = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(qvec, embeddings)[0]
    top_idx = scores.argsort(descending=True)[: min(k, len(df))]
    out = df.iloc[top_idx].copy()
    out["score"] = [float(scores[i]) for i in top_idx]
    return out

try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load {CSV_PATH}: {e}")
    st.stop()

name_col = get_col(df, ["name", "startup", "company"])
category_col = get_col(df, ["category", "focus", "sector", "subcategory", "domain"])
desc_col = get_col(df, ["description", "summary", "about", "overview"])
stage_col = get_col(df, ["stage", "funding stage", "funding_stage"])
location_col = get_col(df, ["hq", "location", "country", "region"])
website_col = get_col(df, ["website", "url", "link"])

required = {"name": name_col, "category": category_col, "description": desc_col}
missing = [k for k, v in required.items() if v is None]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.write("Available columns:", list(df.columns))
    st.stop()

df["search_text"] = (
    df[name_col].map(clean_text) + " " +
    df[category_col].map(clean_text) + " " +
    df[desc_col].map(clean_text) + " " +
    (df[stage_col].map(clean_text) if stage_col else "") + " " +
    (df[location_col].map(clean_text) if location_col else "")
)

location_values = df[location_col].dropna().astype(str).unique().tolist() if location_col else []

keyword_pool = []
for col in [name_col, category_col, desc_col, stage_col]:
    if col:
        keyword_pool.extend(df[col].dropna().astype(str).tolist())

keyword_values = sorted({x.strip() for x in keyword_pool if str(x).strip()})

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me about construction startups, funding stages, regions, categories, or compare companies."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a question about the directory...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    intent = detect_intent(prompt)

    filters = {
        "location_col": location_col,
        "category_col": category_col,
        "desc_col": desc_col,
        "location": detect_location(prompt, location_values),
        "domain_hits": detect_keywords(prompt, keyword_values),
    }

    working_df = apply_text_filters(df, filters)

    reply = ""
    table = None

    if working_df.empty:
        reply = "I couldn't find any matches after applying the filters."
    elif intent == "count":
        matches = top_matches(working_df, prompt, k=min(20, len(working_df)))
        reply = f"I found {len(matches)} relevant startups."
        table = matches[[c for c in [name_col, category_col, stage_col, location_col] if c]].copy()

    elif intent == "stages":
        if stage_col:
            stages = working_df[stage_col].dropna().astype(str).value_counts()
            if stages.empty:
                reply = "No funding stages found in the filtered set."
            else:
                reply = "Funding stages represented: " + ", ".join(stages.index.tolist())
                table = stages.reset_index()
                table.columns = ["stage", "count"]
        else:
            matches = top_matches(working_df, prompt, k=TOP_K)
            reply = "I could not find a dedicated funding stage column, so here are the closest matches."
            table = matches[[c for c in [name_col, category_col, stage_col, location_col] if c]].copy()

    elif intent == "compare":
        names = [n.strip() for n in re.split(r"\b(?:vs|versus|compare)\b|,", prompt, flags=re.I) if n.strip()]
        if len(names) >= 2:
            a, b = names[0], names[1]
            ra = top_matches(df, a, k=1)
            rb = top_matches(df, b, k=1)
            if not ra.empty and not rb.empty:
                reply = f"Here is a quick comparison of {ra.iloc[0][name_col]} and {rb.iloc[0][name_col]}."
                table = pd.DataFrame([
                    {
                        "name": ra.iloc[0][name_col],
                        "category": ra.iloc[0][category_col],
                        "stage": ra.iloc[0][stage_col] if stage_col else "",
                        "location": ra.iloc[0][location_col] if location_col else "",
                        "description": ra.iloc[0][desc_col],
                    },
                    {
                        "name": rb.iloc[0][name_col],
                        "category": rb.iloc[0][category_col],
                        "stage": rb.iloc[0][stage_col] if stage_col else "",
                        "location": rb.iloc[0][location_col] if location_col else "",
                        "description": rb.iloc[0][desc_col],
                    },
                ])
            else:
                reply = "I could not find both companies to compare."
        else:
            matches = top_matches(working_df, prompt, k=TOP_K)
            reply = "Please mention two company names to compare. Here are the closest matches."
            table = matches[[c for c in [name_col, category_col, stage_col, location_col] if c]].copy()

    else:
        matches = top_matches(working_df, prompt, k=TOP_K)
        if matches.empty:
            reply = "I couldn't find a relevant match in the directory."
        else:
            lines = []
            for _, row in matches.iterrows():
                stage_part = f", {row[stage_col]}" if stage_col and pd.notna(row.get(stage_col)) else ""
                lines.append(f"- {row[name_col]} ({row[category_col]}{stage_part}): {row[desc_col]}")
            reply = "Here are the most relevant startups:\n" + "\n".join(lines)
            table = matches[[c for c in [name_col, category_col, stage_col, location_col] if c]].copy()

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.write(reply)
        if table is not None and len(table) > 0:
            st.dataframe(table, use_container_width=True)
