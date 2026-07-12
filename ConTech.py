import re
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title="Construction Startup Directory",
    page_icon="🏗️",
    layout="wide"
)

st.title("BETA ConTech Directory")
st.caption("Search for your ideal ConTech vendor")

CSV_PATH = "data/startups.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

DOMAIN_RULES = {
    "digital": {
        "include": ["digital", "software", "platform", "data", "cloud", "workflow", "automation"],
        "exclude": ["robotics", "robotic", "autonomous robotics", "drone", "inspection", "video analytics"],
    },
    "robotics": {
        "include": ["robotics", "robotic", "autonomous robotics", "automation", "autonomous"],
        "exclude": ["digital", "software", "platform", "cloud", "video analytics"],
    },
    "video analytics": {
        "include": ["video analytics", "camera analytics", "cctv", "computer vision", "video surveillance", "surveillance"],
        "exclude": ["robotics", "robotic", "autonomous robotics", "digital"],
    },
    "site safety": {
        "include": ["site safety", "safety monitoring", "ppe", "hazard detection", "safety alerts", "video analytics", "cctv"],
        "exclude": ["robotics", "robotic", "autonomous robotics", "digital"],
    },
    "sustainability": {
        "include": [
            "sustainability",
            "low carbon",
            "carbon",
            "concrete",
            "cement",
            "material",
            "cooling",
            "wearable",
            "mineralisation",
        ],
        "exclude": [],
    },
}

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


def detect_stage(prompt):
    q = prompt.lower()

    stage_patterns = {
        "pre-seed": r"\bpre[- ]seed\b",
        "seed": r"\bseed\b",
        "series a": r"\bseries\s*a\b",
        "series b": r"\bseries\s*b\b",
        "series c": r"\bseries\s*c\b",
    }

    for stage, pattern in stage_patterns.items():
        if re.search(pattern, q):
            return stage

    return None

def detect_location(prompt, location_values):
    q = prompt.lower()

    for loc in location_values:
        loc = str(loc).strip()
        if re.search(r"\b" + re.escape(loc.lower()) + r"\b", q):
            return loc

    return None

def detect_domain_key(prompt):
    q = prompt.lower()
    for key in DOMAIN_RULES:
        if key in q:
            return key
    return None

def exact_literal_match(series, term):
    return series.astype(str).str.contains(term, case=False, na=False, regex=False)

def apply_domain_rules(df, domain_col, domain_key):
    if not domain_key or not domain_col:
        return df

    rules = DOMAIN_RULES.get(domain_key)
    if not rules:
        return df

    domain_text = df[domain_col].astype(str)

    include_mask = pd.Series(False, index=df.index)
    for term in rules["include"]:
        include_mask |= domain_text.str.contains(term, case=False, na=False, regex=False)

    out = df[include_mask].copy()

    for term in rules["exclude"]:
        out = out[~out[domain_col].astype(str).str.contains(term, case=False, na=False, regex=False)]

    return out

def exact_location_filter(df, location_col, location_value):
    if not location_col or not location_value:
        return df
    return df[df[location_col].astype(str).str.contains(location_value, case=False, na=False, regex=False)]

def build_embeddings_for_df(df):
    model = load_model()
    return model.encode(df["search_text"].tolist(), convert_to_tensor=True)

def semantic_top_matches(df, query, k=TOP_K):
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
category_col = get_col(df, ["category", "focus", "sector", "subcategory"])
desc_col = get_col(df, ["description", "summary", "about", "overview"])
stage_col = get_col(df, ["stage", "funding stage", "funding_stage"])
location_col = get_col(df, ["hq", "location", "country", "region"])
domain_col = get_col(df, ["domain"])
website_col = get_col(df, ["website", "url", "link"])

required = {"name": name_col, "description": desc_col}
missing = [k for k, v in required.items() if v is None]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.write("Available columns:", list(df.columns))
    st.stop()

df["search_text"] = (
    df[name_col].map(clean_text) + " " +
    (df[category_col].map(clean_text) if category_col else "") + " " +
    df[desc_col].map(clean_text) + " " +
    (df[stage_col].map(clean_text) if stage_col else "") + " " +
    (df[location_col].map(clean_text) if location_col else "") + " " +
    (df[domain_col].map(clean_text) if domain_col else "")
)


location_values = df[location_col].dropna().astype(str).unique().tolist() if location_col else []

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
    domain_key = detect_domain_key(prompt)
    stage_key = detect_stage(prompt)
    location_value = detect_location(prompt, location_values)



    filtered = df.copy()


    filtered = apply_domain_rules(df, domain_col, domain_key)

    if domain_key:
        filtered = filtered[
            filtered["Domain"].str.lower() == domain_key.lower()
        ]
  
    if stage_key:
        filtered = filtered[
            filtered["Funding Stage"].str.lower() == stage_key.lower()
        ]


    if location_value:
        filtered = filtered[
            filtered["Location"].str.lower() == location_value.lower()
        ]

    
    st.write(filtered[[name_col, desc_col]])

    reply = ""
    table = None


    if filtered.empty:
        reply = "I couldn't find any matches after applying the strict filters."

    elif intent == "count":
        if domain_key:
            reply = f"I found {len(filtered)} startups matching {domain_key}."
        elif location_value:
            reply = f"I found {len(filtered)} startups based in {location_value}."
        else:
            reply = f"I found {len(filtered)} relevant startups."
        table = filtered[[c for c in [name_col, category_col, stage_col, location_col, domain_col] if c]].copy()

    elif intent == "stages":
        if stage_col:
            stages = filtered[stage_col].dropna().astype(str).value_counts()
            if stages.empty:
                reply = "No funding stages found in the filtered set."
            else:
                reply = "Funding stages represented: " + ", ".join(stages.index.tolist())
                table = stages.reset_index()
                table.columns = ["stage", "count"]
        else:
            reply = "I could not find a dedicated funding stage column."
            table = filtered[[c for c in [name_col, category_col, stage_col, location_col, domain_col] if c]].copy()

    elif intent == "compare":
        names = [n.strip() for n in re.split(r"\b(?:vs|versus|compare)\b|,", prompt, flags=re.I) if n.strip()]
        if len(names) >= 2:
            a, b = names[0], names[1]
            ra = semantic_top_matches(df, a, k=1)
            rb = semantic_top_matches(df, b, k=1)
            if not ra.empty and not rb.empty:
                reply = f"Here is a quick comparison of {ra.iloc[0][name_col]} and {rb.iloc[0][name_col]}."
                table = pd.DataFrame([
                    {
                        "name": ra.iloc[0][name_col],
                        "category": ra.iloc[0][category_col] if category_col else "",
                        "stage": ra.iloc[0][stage_col] if stage_col else "",
                        "location": ra.iloc[0][location_col] if location_col else "",
                        "domain": ra.iloc[0][domain_col] if domain_col else "",
                        "description": ra.iloc[0][desc_col],
                    },
                    {
                        "name": rb.iloc[0][name_col],
                        "category": rb.iloc[0][category_col] if category_col else "",
                        "stage": rb.iloc[0][stage_col] if stage_col else "",
                        "location": rb.iloc[0][location_col] if location_col else "",
                        "domain": rb.iloc[0][domain_col] if domain_col else "",
                        "description": rb.iloc[0][desc_col],
                    },
                ])
            else:
                reply = "I could not find both companies to compare."
        else:
            reply = "Please mention two company names to compare."
            table = filtered[[c for c in [name_col, category_col, stage_col, location_col, domain_col] if c]].copy()

    else:
        if domain_key or location_value:
            reply = "Here are the matching startups:\n" + "\n".join(
                f"- {row[name_col]} ({row[domain_col] if domain_col else row[category_col]}{', ' + str(row[stage_col]) if stage_col and pd.notna(row.get(stage_col)) else ''}): {row[desc_col]}"
                for _, row in filtered.iterrows()
            )
            table = filtered[[c for c in [name_col, category_col, stage_col, location_col, domain_col] if c]].copy()
        else:
            matches = semantic_top_matches(filtered, prompt, k=TOP_K)
            if matches.empty:
                reply = "I couldn't find a relevant match in the directory."
            else:
                lines = []
                for _, row in matches.iterrows():
                    stage_val = row[stage_col] if stage_col and pd.notna(row.get(stage_col)) else None
                    stage_part = f", {stage_val}" if stage_val is not None else ""
                    lines.append(f"- {row[name_col]} ({row[domain_col] if domain_col else row[category_col]}{stage_part}): {row[desc_col]}")
                reply = "Here are the most relevant startups:\n" + "\n".join(lines)
                table = matches[[c for c in [name_col, category_col, stage_col, location_col, domain_col] if c]].copy()

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.write(reply)
         if table is not None and len(table) > 0:
                    st.dataframe(table,use_container_width=True,hide_index=True)
