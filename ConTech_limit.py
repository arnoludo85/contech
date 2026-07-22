import re
from difflib import SequenceMatcher

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util


st.set_page_config(
    page_title="Construction Startup Directory",
    page_icon="🏗️",
    layout="wide"
)

st.title("BETA ConTech Directory")
st.caption("Search for your ConTech vendor")

CSV_PATH = "data/startups.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
RELEVANCE_THRESHOLD = 0.75
DOMAIN_MATCH_THRESHOLD = 0.84


DOMAIN_RULES = {
    "digital": {
        "include": ["digital", "software", "platform", "data", "cloud", "workflow", "automation"],
        "aliases": ["digital tools", "software platform", "workflow automation", "cloud based"],
        "exclude": ["robotics", "robotic", "autonomous robotics", "drone", "inspection", "video analytics"],
    },
    "robotics": {
        "include": ["robotics", "robotic", "autonomous robotics", "automation", "autonomous"],
        "aliases": ["autonomous robot", "robot", "robotic system", "autonomous machine"],
        "exclude": ["digital", "software", "platform", "cloud", "video analytics"],
    },
    "video analytics": {
        "include": ["video analytics", "camera analytics", "cctv", "computer vision", "video surveillance", "surveillance"],
        "aliases": ["vision analytics", "camera vision", "ai video", "video monitoring"],
        "exclude": ["robotics", "robotic", "autonomous robotics", "digital"],
    },
    "site safety": {
        "include": ["site safety", "safety monitoring", "ppe", "hazard detection", "safety alerts", "video analytics", "cctv"],
        "aliases": ["worker safety", "construction safety", "safety tech", "jobsite safety"],
        "exclude": ["robotics", "robotic", "autonomous robotics", "digital"],
    },
    "sustainability": {
        "include": [
            "sustainability",
            "sustainable",
            "low carbon",
            "carbon",
            "concrete",
            "cement",
            "material",
            "cooling",
            "wearable",
            "mineralisation",
            "mineralization",
            "decarbonization",
            "decarbonisation",
            "net zero",
            "circular",
            "recycled",
        ],
        "aliases": [
            "green",
            "low carbon materials",
            "carbon reduction",
            "carbon negative",
            "climate tech",
            "eco friendly",
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


def normalize_text(x):
    x = clean_text(x).lower()
    x = re.sub(r"[^a-z0-9\s]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def fuzzy_term_in_text(text, term, threshold=DOMAIN_MATCH_THRESHOLD):
    """
    Returns True for:
    - exact substring matches
    - close matches like 'robotic' vs 'robotics'
    - slight phrase variations like 'autonomous robot' vs 'autonomous robotics'
    """
    text_n = normalize_text(text)
    term_n = normalize_text(term)

    if not text_n or not term_n:
        return False

    if term_n in text_n:
        return True

    # Whole-text similarity
    if SequenceMatcher(None, text_n, term_n).ratio() >= threshold:
        return True

    text_tokens = text_n.split()
    term_tokens = term_n.split()

    # Single-word terms: compare against each token
    if len(term_tokens) == 1:
        if any(SequenceMatcher(None, term_n, tok).ratio() >= threshold for tok in text_tokens):
            return True

    # Multi-word terms: compare against sliding windows of nearby lengths
    window_sizes = sorted(set([max(1, len(term_tokens) - 1), len(term_tokens), len(term_tokens) + 1]))
    for window_size in window_sizes:
        if window_size > len(text_tokens):
            continue
        for i in range(len(text_tokens) - window_size + 1):
            window = " ".join(text_tokens[i:i + window_size])
            if SequenceMatcher(None, term_n, window).ratio() >= threshold:
                return True

    return False


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


def detect_domain_keys(prompt):
    """
    Fuzzy domain detection:
    - catches typos / near-matches
    - catches aliases and softer wording
    - works for queries like 'robotic', 'sustainable', 'autonomous robotics'
    """
    matched = []
    for key, rules in DOMAIN_RULES.items():
        candidates = [key] + rules.get("include", []) + rules.get("aliases", [])
        if any(fuzzy_term_in_text(prompt, term) for term in candidates):
            matched.append(key)
    return matched


def detect_compare_companies(prompt, df, name_col):
    q = prompt.lower()
    companies = []
    for company in df[name_col].dropna():
        if company.lower() in q:
            companies.append(company)
    return companies


def exact_literal_match(series, term):
    return series.astype(str).str.contains(term, case=False, na=False, regex=False)


def contains_literal(series, term):
    return series.astype(str).str.contains(term, case=False, regex=False, na=False)


def apply_domain_rules(df, domain_col, domain_keys):
    """
    Keep rows that match any domain's inclusion terms.
    This is intentionally broader than exact equality so that:
    - query typos still work via detect_domain_keys()
    - description wording variations still match
    """
    if not domain_keys:
        return df

    include_mask = pd.Series(False, index=df.index)

    for domain in domain_keys:
        rules = DOMAIN_RULES.get(domain)
        if not rules:
            continue

        domain_mask = pd.Series(False, index=df.index)

        search_terms = rules["include"] + rules.get("aliases", [])

        for term in search_terms:
            mask = (
                contains_literal(df[domain_col], term) if domain_col else pd.Series(False, index=df.index)
            ) | contains_literal(df["Description"], term) | contains_literal(df["search_text"], term)

            domain_mask |= mask

        include_mask |= domain_mask

    return df[include_mask]


def exact_location_filter(df, location_col, location_value):
    if not location_col or not location_value:
        return df
    return df[df[location_col].astype(str).str.contains(location_value, case=False, na=False, regex=False)]


def build_embeddings_for_df(df):
    model = load_model()
    return model.encode(df["search_text"].tolist(), convert_to_tensor=True)


def semantic_top_matches(df, query, k=TOP_K):
    if df.empty:
        empty = df.copy()
        empty["score"] = []
        return empty

    model = load_model()
    embeddings = build_embeddings_for_df(df)
    qvec = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(qvec, embeddings)[0]
    top_idx = scores.argsort(descending=True)[: min(k, len(df))]
    out = df.iloc[top_idx].copy()
    out["score"] = [float(scores[i]) for i in top_idx]
    return out


def should_reask_user(intent, filtered_df, prompt, domain_keys, location_value):
    """
    Ask the user to rewrite the query when the search is not strong enough.
    Only applies to normal search queries, not count/compare/stages.
    """
    if intent != "search":
        return False, 1.0, pd.DataFrame()

    if filtered_df.empty:
        return True, 0.0, pd.DataFrame()

    matches = semantic_top_matches(filtered_df, prompt, k=TOP_K)
    best_score = float(matches["score"].max()) if not matches.empty else 0.0

    # If the user already specified a structured filter, let that route handle the response.
    if domain_keys or location_value:
        return False, best_score, matches

    return best_score < RELEVANCE_THRESHOLD, best_score, matches


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
    domain_keys = detect_domain_keys(prompt)
    stage_key = detect_stage(prompt)
    location_value = detect_location(prompt, location_values)

    filtered = df.copy()
    filtered = apply_domain_rules(df, domain_col, domain_keys)

    # Keep these filters flexible by using the detected column names
    # IMPORTANT CHANGE:
    # Do NOT require the domain column to equal the domain keyword exactly.
    # That made queries like "robotic" or "sustainable" too brittle.
    if stage_key and stage_col:
        filtered = filtered[
            filtered[stage_col]
            .astype(str)
            .str.strip()
            .str.lower()
            == stage_key.lower()
        ]

    if location_value and location_col:
        filtered = filtered[
            filtered[location_col]
            .astype(str)
            .str.strip()
            .str.lower()
            == location_value.lower()
        ]

    reply = ""
    table = None

    reask_user, best_score, matches = should_reask_user(
        intent=intent,
        filtered_df=filtered,
        prompt=prompt,
        domain_keys=domain_keys,
        location_value=location_value,
    )

    if reask_user:
        reply = (
            "I could not find a strong match in the database. "
            f"The best relevance score was {best_score:.0%}, which is below the 75% threshold. "
            "Please post your query again with a more relevant and specific question about the startups database "
            "(for example: a company name, funding stage, location, category, or domain)."
        )

    elif filtered.empty:
        reply = (
            "I couldn't find any matches after applying the available filters. "
            "Please post your query again with a more relevant question."
        )

    elif intent == "count":
        if domain_keys:
            reply = f"I found {len(filtered)} startups matching {domain_keys}."
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
        companies = detect_compare_companies(prompt, df, name_col)

        if len(companies) < 2:
            reply = "Please mention two company names to compare."
        else:
            comparison = (
                df[df[name_col].isin(companies)][[
                    name_col,
                    domain_col,
                    stage_col,
                    desc_col,
                    location_col,
                    website_col
                ]]
                .set_index(name_col)
                .T
            )
            comparison.index.name = "Attribute"
            comparison = comparison.reset_index()

            reply = f"Comparison between {companies[0]} and {companies[1]}."
            table = comparison

    else:
        if domain_keys or location_value:
            reply = "Here are the matching startups:\n" + "\n".join(
                f"- {row[name_col]} ({row[domain_col] if domain_col else row[category_col]}"
                f"{', ' + str(row[stage_col]) if stage_col and pd.notna(row.get(stage_col)) else ''}): {row[desc_col]}"
                for _, row in filtered.iterrows()
            )
            table = filtered[[c for c in [name_col, category_col, stage_col, location_col, domain_col] if c]].copy()
        else:
            # Only answer if the semantic confidence is high enough
            matches = semantic_top_matches(filtered, prompt, k=TOP_K)
            top_score = float(matches["score"].max()) if not matches.empty else 0.0

            if top_score < RELEVANCE_THRESHOLD:
                reply = (
                    "I couldn't find a strong enough match in the database. "
                    f"The top relevance score was {top_score:.0%}, below the 75% threshold. "
                    "Please post your query again with a more relevant question about the directory."
                )
            else:
                lines = []
                for _, row in matches.iterrows():
                    stage_val = row[stage_col] if stage_col and pd.notna(row.get(stage_col)) else None
                    stage_part = f", {stage_val}" if stage_val is not None else ""

                    lines.append(
                        f"- {row[name_col]} ({row[domain_col] if domain_col else row[category_col]}{stage_part}): {row[desc_col]}"
                    )

                reply = "Here are the most relevant startups:\n" + "\n".join(lines)
                table = matches[[c for c in [name_col, category_col, stage_col, location_col, domain_col] if c]].copy()

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.write(reply)
        if table is not None and len(table) > 0:
            st.dataframe(table, use_container_width=True, hide_index=True)
