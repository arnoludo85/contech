import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Optional, Set, List

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Construction Startup Directory", page_icon="🏗️", layout="wide")

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX = APP_DIR / "AI database_trial.xlsx"
PRIMARY_SHEET = "Vendor List To Share"

DISPLAY_COLUMNS = [
    "Vendor Name",
    "Type of Solution",
    "AI Use Cases",
    "Country of Origin",
    "Have a local entity or reseller?",
    "PSG Status",
    "Funding Stage",
    "Company Size (No. of Employees)",
    "Website",
]

ALLOW_FUZZY_FALLBACK = False
MIN_RELEVANCE = 0.25

COUNTRY_ALIASES = {
    "singapore": "Singapore",
    "usa": "USA",
    "u.s.a": "USA",
    "us": "USA",
    "united states": "USA",
    "uk": "UK",
    "united kingdom": "UK",
    "australia": "Australia",
    "france": "France",
    "germany": "Germany",
    "netherlands": "Netherlands",
    "belgium": "Belgium",
    "hong kong": "Hong Kong",
    "china": "China",
    "japan": "Japan",
    "south korea": "South Korea",
    "canada": "Canada",
    "india": "India",
    "malaysia": "Malaysia",
    "indonesia": "Indonesia",
    "thailand": "Thailand",
    "vietnam": "Vietnam",
    "ireland": "Ireland",
    "new york": "New York",
    "munich": "Munich",
    "dublin": "Dublin",
}

USE_CASE_ALIASES = {
    "code compliance": ["Detailed Design & Code Compliance Review"],
    "detailed design and code compliance review": ["Detailed Design & Code Compliance Review"],
    "site documentation": ["Site Documentation & Knowledge Management"],
    "site documentation and knowledge management": ["Site Documentation & Knowledge Management"],
    "knowledge management": ["Site Documentation & Knowledge Management"],
    "site monitoring": ["AI-Enabled Site Monitoring"],
    "progress analytics": ["Reality Capture & Progress Analytics"],
    "reality capture": ["Reality Capture & Progress Analytics", "Drone; Reality Capture (scanner)"],
    "schedule optimisation": ["Schedule Optimisation & Predictive Analytics"],
    "schedule optimization": ["Schedule Optimisation & Predictive Analytics"],
    "predictive analytics": ["Schedule Optimisation & Predictive Analytics"],
    "feasibility design": ["AI-Assisted Feasibility Design & Visualisation"],
    "visualisation": ["AI-Assisted Feasibility Design & Visualisation"],
    "visualization": ["AI-Assisted Feasibility Design & Visualisation"],
    "bim automation": ["BIM Workflow Support / Automation"],
    "tender": ["Tender Preparation, Intelligence & Evaluation"],
    "quantity take-off": ["Quantity Take-off"],
    "quantity takeoff": ["Quantity Take-off"],
    "carbon accounting": ["Carbon Accounting", "Corporate Carbon Accounting"],
}

INTENT_WORDS = {
    "recommend",
    "recommendation",
    "suggest",
    "suggestion",
    "find",
    "show",
    "list",
    "display",
    "best",
    "top",
    "please",
}

LIST_ALL_PATTERNS = [
    r"\blist\b.*\ball\b",
    r"\bshow\b.*\ball\b",
    r"\bdisplay\b.*\ball\b",
    r"\ball\b.*\bstartups\b",
    r"\bevery\b.*\bstartup\b",
    r"\bfull\b.*\bdirectory\b",
    r"\bcomplete\b.*\bdirectory\b",
]

COUNT_PATTERNS = [
    r"\bhow many\b",
    r"\bcount\b",
    r"\bnumber of\b",
]

STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "on", "for", "a", "an", "with", "by",
    "startup", "startups", "company", "companies", "directory", "based",
    "are", "is", "there", "from", "this", "that", "those", "these", "part",
    "which", "what", "who", "whom", "whose", "me", "you", "us", "our", "your",
    "not", "all", "how", "many", "count",
    "overseas", "local", "entity", "reseller",
    "any", "exist", "exists", "available", "can", "could",
}.union(INTENT_WORDS)

QUERY_HINTS = [
    "Which startups are based in Singapore?",
    "Are there any overseas startups?",
    "Are there any VR startups?",
    "Which startups have code compliance as a use case?",
    "Recommend digital twin VR startups",
]

# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------
@dataclass
class QueryPlan:
    raw_query: str
    normalized_query: str
    intent: str  # list | count | recommend | search
    company_name: Optional[str] = None
    include_countries: Set[str] = field(default_factory=set)
    exclude_countries: Set[str] = field(default_factory=set)
    funding_stages: List[str] = field(default_factory=list)
    psg_status: Optional[str] = None  # yes | no | None
    local_entity: Optional[str] = None  # yes | no | None
    solution_terms: List[str] = field(default_factory=list)
    use_case_terms: List[str] = field(default_factory=list)
    free_terms: List[str] = field(default_factory=list)

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "0":
        return ""
    return re.sub(r"\s+", " ", s).lower()


def normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"[^\w\s&/.-]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q


def normalize_query_terms(q: str) -> str:
    q = normalize_query(q)
    q = q.replace("-", " ")
    q = re.sub(r"\bvirtual reality\b", "ar/vr", q)
    q = re.sub(r"\bar\s*[/\s-]\s*vr\b", "ar/vr", q)
    q = re.sub(r"(?<!ar/)\bvr\b", "ar/vr", q)
    return q


def split_phrases(value) -> list[str]:
    if pd.isna(value):
        return []
    parts = re.split(r"[;,/|]", str(value))
    return [p.strip().lower() for p in parts if p.strip()]


def unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def row_blob(row: pd.Series) -> str:
    fields = [
        row.get("Vendor Name", ""),
        row.get("Type of Solution", ""),
        row.get("AI Use Cases", ""),
        row.get("Country of Origin", ""),
        row.get("Funding Stage", ""),
        row.get("PSG Status", ""),
        row.get("Have a local entity or reseller?", ""),
        row.get("Company Size (No. of Employees)", ""),
        row.get("Website", ""),
    ]
    return " | ".join(normalize_text(v) for v in fields)


def row_contains_any_terms(row: pd.Series, terms: list[str]) -> bool:
    if not terms:
        return False
    blob = row_blob(row)
    return any(term.lower() in blob for term in terms)


def row_contains_all_terms(row: pd.Series, terms: list[str]) -> bool:
    if not terms:
        return True
    blob = row_blob(row)
    return all(term.lower() in blob for term in terms)

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_directory(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        source = uploaded_file
    else:
        if not DEFAULT_XLSX.exists():
            raise FileNotFoundError(f"Workbook not found: {DEFAULT_XLSX}. Upload the Excel file in the sidebar.")
        source = str(DEFAULT_XLSX)

    xls = pd.ExcelFile(source)
    sheet = PRIMARY_SHEET if PRIMARY_SHEET in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(source, sheet_name=sheet)

    df.columns = [str(c).strip() for c in df.columns]

    for col in DISPLAY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace({0: np.nan, "0": np.nan, "": np.nan, "nan": np.nan})

    search_cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    df["_search_blob"] = df[search_cols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
    return df

# -------------------------------------------------------------------
# Parsing helpers
# -------------------------------------------------------------------
def is_list_all_query(q: str) -> bool:
    q = normalize_query(q)
    return any(re.search(p, q) for p in LIST_ALL_PATTERNS)


def is_count_query(q: str) -> bool:
    q = normalize_query(q)
    return any(re.search(p, q) for p in COUNT_PATTERNS)


def is_recommendation_query(q: str) -> bool:
    q = normalize_query(q)
    words = set(q.split())
    return bool(words.intersection({"recommend", "recommendation", "suggest", "suggestion"}))


def detect_company_name(query: str, df: pd.DataFrame):
    if "Vendor Name" not in df.columns or df.empty:
        return None, 0.0

    q = query.strip().lower()
    names = df["Vendor Name"].dropna().astype(str).tolist()

    exact = [n for n in names if n.lower() == q]
    if exact:
        return exact[0], 1.0

    close = get_close_matches(query.strip(), names, n=1, cutoff=0.80)
    if close:
        return close[0], 0.95

    for name in names:
        n = name.lower()
        if q and (q in n or n in q):
            return name, 0.85

    return None, 0.0


def get_known_solution_values(df: pd.DataFrame) -> set[str]:
    values = set()
    if "Type of Solution" in df.columns:
        for v in df["Type of Solution"].dropna().astype(str):
            values.update(split_phrases(v))
            values.add(v.lower())
    return {v for v in values if v}


def get_known_use_case_values(df: pd.DataFrame) -> set[str]:
    values = set()
    if "AI Use Cases" in df.columns:
        for v in df["AI Use Cases"].dropna().astype(str):
            values.update(split_phrases(v))
            values.add(v.lower())
    values.update(USE_CASE_ALIASES.keys())
    return {v for v in values if v}


def extract_country_constraints(query: str):
    q = normalize_query_terms(query)
    include = set()
    exclude = set()

    if (
        re.search(r"\boverseas\b", q)
        or re.search(r"\babroad\b", q)
        or re.search(r"\binternational\b", q)
        or re.search(r"\boutside singapore\b", q)
        or re.search(r"\bnon[-\s]?singapore\b", q)
        or re.search(r"\bnot based in singapore\b", q)
        or re.search(r"\boverseas based\b", q)
    ):
        exclude.add("Singapore")

    if (
        re.search(r"\bsingapore[-\s]?based\b", q)
        or re.search(r"\bbased in singapore\b", q)
        or re.search(r"\bin singapore\b", q)
    ):
        include.add("Singapore")

    for alias, canon in COUNTRY_ALIASES.items():
        pat = re.escape(alias)

        if (
            re.search(rf"\bnot based in\s+{pat}\b", q)
            or re.search(rf"\bnot in\s+{pat}\b", q)
            or re.search(rf"\boutside\s+{pat}\b", q)
            or re.search(rf"\bexcluding\s+{pat}\b", q)
            or re.search(rf"\bnon[-\s]+{pat}\b", q)
        ):
            exclude.add(canon)
        elif (
            re.search(rf"\bbased in\s+{pat}\b", q)
            or re.search(rf"\b{pat}[-\s]?based\b", q)
            or re.search(rf"\bin\s+{pat}\b", q)
        ):
            include.add(canon)

    return include, exclude


def extract_funding_stages(query: str) -> list[str]:
    q = normalize_query_terms(query)
    if "pre-seed" in q or "pre seed" in q:
        return ["pre-seed", "pre seed"]
    if re.search(r"\bseed\b", q):
        return ["Seed Round", "seed round", "Seed", "Early stage"]
    if "series a" in q:
        return ["Series A"]
    if "series b" in q:
        return ["Series B"]
    if "series c" in q:
        return ["Series C"]
    if "ipo" in q:
        return ["IPO"]
    if "acquired" in q:
        return ["Acquired"]
    return []


def extract_boolean_filters(query: str):
    q = normalize_query_terms(query)

    psg_status = None
    local_entity = None

    if "psg" in q:
        psg_status = "no" if re.search(r"\b(no|not|non|without)\b", q) else "yes"

    if "local entity" in q or "reseller" in q:
        local_entity = "no" if re.search(r"\b(no|not|non|without)\b", q) else "yes"

    return psg_status, local_entity


def extract_exact_use_cases(query: str, df: pd.DataFrame) -> list[str]:
    q = normalize_query_terms(query)
    canonical = []

    for alias, mapped_values in USE_CASE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            canonical.extend(mapped_values)

    known_use_cases = get_known_use_case_values(df)
    for val in known_use_cases:
        if len(val) >= 4 and val in q:
            canonical.append(val)

    return unique_preserve_order(canonical)


def extract_exact_solution_terms(query: str, df: pd.DataFrame) -> list[str]:
    q = normalize_query_terms(query)
    canonical = []

    known_solutions = get_known_solution_values(df)
    for val in known_solutions:
        if len(val) >= 3 and val in q:
            canonical.append(val)

    return unique_preserve_order(canonical)


def extract_free_terms(query: str, used_terms: list[str]) -> list[str]:
    q = normalize_query_terms(query)
    tokens = re.findall(r"[a-z0-9/]+", q)

    used = set()
    for term in used_terms:
        for token in re.findall(r"[a-z0-9/]+", term):
            used.add(token)

    free = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) < 3:
            continue
        if token in used:
            continue
        free.append(token)

    return unique_preserve_order(free)

# -------------------------------------------------------------------
# Query parser
# -------------------------------------------------------------------
def parse_query(query: str, df: pd.DataFrame) -> QueryPlan:
    raw = query or ""
    normalized = normalize_query_terms(raw)

    if is_list_all_query(normalized):
        intent = "list"
    elif is_count_query(normalized):
        intent = "count"
    elif is_recommendation_query(normalized):
        intent = "recommend"
    else:
        intent = "search"

    solution_terms = extract_exact_solution_terms(normalized, df)
    use_case_terms = extract_exact_use_cases(normalized, df)
    used_terms = solution_terms + use_case_terms
    free_terms = extract_free_terms(normalized, used_terms)

    include_countries, exclude_countries = extract_country_constraints(normalized)
    funding_stages = extract_funding_stages(normalized)
    psg_status, local_entity = extract_boolean_filters(normalized)

    return QueryPlan(
        raw_query=raw,
        normalized_query=normalized,
        intent=intent,
        include_countries=include_countries,
        exclude_countries=exclude_countries,
        funding_stages=funding_stages,
        psg_status=psg_status,
        local_entity=local_entity,
        solution_terms=solution_terms,
        use_case_terms=use_case_terms,
        free_terms=free_terms,
    )

# -------------------------------------------------------------------
# Filtering pipeline
# -------------------------------------------------------------------
def apply_country_filters(df: pd.DataFrame, include: Set[str], exclude: Set[str]) -> pd.DataFrame:
    out = df.copy()
    if "Country of Origin" not in out.columns:
        return out

    country_col = out["Country of Origin"].astype(str).str.strip()

    if include:
        include_mask = pd.Series(False, index=out.index)
        for country in include:
            include_mask = include_mask | country_col.str.contains(re.escape(country), case=False, na=False, regex=True)
        out = out[include_mask]

    if exclude:
        exclude_mask = pd.Series(False, index=out.index)
        for country in exclude:
            exclude_mask = exclude_mask | country_col.str.contains(re.escape(country), case=False, na=False, regex=True)
        out = out[~exclude_mask]

        if not include:
            out = out[out["Country of Origin"].astype(str).str.strip().ne("")]

    return out


def apply_funding_stage_filters(df: pd.DataFrame, stages: list[str]) -> pd.DataFrame:
    out = df.copy()
    if not stages or "Funding Stage" not in out.columns:
        return out

    mask = pd.Series(False, index=out.index)
    for stage in stages:
        mask = mask | out["Funding Stage"].astype(str).str.contains(re.escape(stage), case=False, na=False, regex=True)
    return out[mask]


def apply_boolean_filters(df: pd.DataFrame, psg_status: Optional[str], local_entity: Optional[str]) -> pd.DataFrame:
    out = df.copy()

    if psg_status is not None and "PSG Status" in out.columns:
        if psg_status == "yes":
            out = out[out["PSG Status"].astype(str).str.contains(r"\byes\b", case=False, na=False, regex=True)]
        else:
            out = out[out["PSG Status"].astype(str).str.contains(r"\bno\b", case=False, na=False, regex=True)]

    if local_entity is not None and "Have a local entity or reseller?" in out.columns:
        if local_entity == "yes":
            out = out[out["Have a local entity or reseller?"].astype(str).str.contains(r"\byes\b", case=False, na=False, regex=True)]
        else:
            out = out[out["Have a local entity or reseller?"].astype(str).str.contains(r"\bno\b", case=False, na=False, regex=True)]

    return out


def apply_content_filter(df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    if not terms:
        return df
    return df[df.apply(lambda row: row_contains_any_terms(row, terms), axis=1)]


def apply_strict_content_filter(df: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    out = df.copy()
    out = apply_country_filters(out, plan.include_countries, plan.exclude_countries)
    out = apply_funding_stage_filters(out, plan.funding_stages)
    out = apply_boolean_filters(out, plan.psg_status, plan.local_entity)

    content_terms = unique_preserve_order(plan.solution_terms + plan.use_case_terms + plan.free_terms)

    # Geography-only queries should not require content words.
    if content_terms:
        out = apply_content_filter(out, content_terms)

    return out

# -------------------------------------------------------------------
# Ranking layer
# -------------------------------------------------------------------
def rank_rows(df: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    if df.empty:
        return df

    solution_terms = plan.solution_terms
    use_case_terms = plan.use_case_terms
    free_terms = plan.free_terms

    rows = []
    for _, row in df.iterrows():
        blob = row_blob(row)
        vendor_text = normalize_text(row.get("Vendor Name", ""))
        solution_text = normalize_text(row.get("Type of Solution", ""))
        usecase_text = normalize_text(row.get("AI Use Cases", ""))
        country_text = normalize_text(row.get("Country of Origin", ""))
        funding_text = normalize_text(row.get("Funding Stage", ""))

        score = 0.0
        field_hits = set()

        for term in solution_terms:
            if term in solution_text:
                score += 4.0
                field_hits.add("solution")
            elif term in blob:
                score += 1.5
                field_hits.add("blob")

        for term in use_case_terms:
            if term in usecase_text:
                score += 4.0
                field_hits.add("usecase")
            elif term in blob:
                score += 1.5
                field_hits.add("blob")

        for term in free_terms:
            if term in blob:
                score += 0.5

        if "solution" in field_hits and "usecase" in field_hits:
            score += 2.0

        if any(term in vendor_text for term in (solution_terms + use_case_terms + free_terms)):
            score += 1.0

        if plan.include_countries and any(c.lower() in country_text for c in plan.include_countries):
            score += 1.0
        if plan.funding_stages and any(s.lower() in funding_text for s in plan.funding_stages):
            score += 0.5

        rows.append(score)

    ranked = df.copy()
    ranked["_score"] = rows
    ranked = ranked[ranked["_score"] > 0].sort_values("_score", ascending=False)
    return ranked

# -------------------------------------------------------------------
# Answering
# -------------------------------------------------------------------
def clarification_message(query: str) -> str:
    return (
        f"I could not confidently match **{query}** to this directory.\n\n"
        "Try one of these:\n"
        "- a company name\n"
        "- a country, such as Singapore\n"
        "- a solution type, such as robotics or digital twin\n"
        "- an AI use case, such as code compliance or site documentation\n"
        "- a funding stage, such as seed round or series A\n"
        "- list all startups in the directory\n"
        "- recommend digital twin VR startups"
    )


def query_is_relevant(plan: QueryPlan) -> bool:
    if plan.intent in {"list", "count", "recommend"}:
        return True
    return bool(
        plan.include_countries
        or plan.exclude_countries
        or plan.funding_stages
        or plan.solution_terms
        or plan.use_case_terms
    )


def detect_company_lookup(query: str, df: pd.DataFrame):
    company, confidence = detect_company_name(query, df)
    if not company:
        return None, 0.0

    matched = df[df["Vendor Name"].astype(str).str.lower() == company.lower()].copy()
    return matched, confidence


def answer_query(query: str, df: pd.DataFrame):
    plan = parse_query(query, df)

    company_matches, company_conf = detect_company_lookup(query, df)
    if company_matches is not None and not company_matches.empty:
        return {
            "type": "company_lookup",
            "message": f"Yes — **{company_matches.iloc[0]['Vendor Name']}** is in the directory.",
            "data": company_matches,
            "confidence": company_conf,
        }

    if plan.intent == "list":
        return {
            "type": "list",
            "message": f"I found **{len(df)}** record(s) in the directory.",
            "data": df.copy(),
            "confidence": 1.0,
        }

    if plan.intent == "recommend":
        hard_filtered = apply_country_filters(df, plan.include_countries, plan.exclude_countries)
        hard_filtered = apply_funding_stage_filters(hard_filtered, plan.funding_stages)
        hard_filtered = apply_boolean_filters(hard_filtered, plan.psg_status, plan.local_entity)

        ranked = rank_rows(hard_filtered, plan)

        if ranked.empty:
            return {
                "type": "clarify",
                "message": clarification_message(query),
                "data": pd.DataFrame(),
                "confidence": 0.0,
            }

        return {
            "type": "recommendation",
            "message": f"I found **{len(ranked)}** recommended match(es) for your query.",
            "data": ranked.head(20).copy(),
            "confidence": float(ranked["_score"].max()) if "_score" in ranked.columns else 1.0,
        }

    filtered = apply_strict_content_filter(df, plan)

    if plan.intent == "count":
        return {
            "type": "count",
            "message": f"I found **{len(filtered)}** matching record(s).",
            "data": filtered.copy(),
            "confidence": 1.0 if len(filtered) else 0.4,
        }

    if not filtered.empty and query_is_relevant(plan):
        ranked = rank_rows(filtered, plan)
        if not ranked.empty:
            return {
                "type": "search",
                "message": f"I found **{len(filtered)}** exact match(es) for your query.",
                "data": ranked.head(50).copy(),
                "confidence": 1.0,
            }
        return {
            "type": "search",
            "message": f"I found **{len(filtered)}** exact match(es) for your query.",
            "data": filtered.copy(),
            "confidence": 1.0,
        }

    if not query_is_relevant(plan):
        return {
            "type": "clarify",
            "message": clarification_message(query),
            "data": pd.DataFrame(),
            "confidence": 0.0,
        }

    return {
        "type": "clarify",
        "message": clarification_message(query),
        "data": pd.DataFrame(),
        "confidence": 0.0,
    }

# -------------------------------------------------------------------
# UI helpers
# -------------------------------------------------------------------
def render_error(msg: str):
    st.error(msg)
    st.info("Check the terminal output for the exact error if the file still cannot load.")


def list_output(df: pd.DataFrame, limit: int = 500):
    cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    out = df[cols].copy()
    if "_score" in df.columns:
        out.insert(0, "Score", df["_score"].round(3).values)
    return out.head(limit)

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
def main():
    st.title("🏗️ Construction Startup Directory")
    st.caption("Search the workbook conversationally and browse the records.")

    with st.sidebar:
        st.header("Data source")
        uploaded = st.file_uploader("Upload the Excel file", type=["xlsx"])
        use_default = st.checkbox("Use local workbook", value=(uploaded is None))

    if uploaded is None and not use_default:
        st.warning("Upload the Excel file or enable the local workbook.")
        st.stop()

    try:
        df = load_directory(uploaded if uploaded is not None else None)
    except Exception as e:
        render_error(f"Could not load the workbook: {e}")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "last_results" not in st.session_state:
        st.session_state.last_results = pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(df))
    c2.metric(
        "Countries",
        int(df["Country of Origin"].replace("", np.nan).nunique(dropna=True)) if "Country of Origin" in df.columns else 0,
    )
    c3.metric(
        "PSG Yes",
        int(df["PSG Status"].astype(str).str.contains("yes", case=False, na=False).sum()) if "PSG Status" in df.columns else 0,
    )
    c4.metric(
        "Named companies",
        int(df["Vendor Name"].replace("", np.nan).nunique(dropna=True)) if "Vendor Name" in df.columns else 0,
    )

    tab_chat, tab_table = st.tabs(["Chatbot search", "Directory table"])

    with tab_chat:
        st.subheader("Ask a question")

        if not st.session_state.chat_history:
            st.info(
                "Try: Which startups are based in Singapore? / Are there any overseas startups? / "
                "Are there any VR startups? / Recommend digital twin VR startups"
            )

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("table") is not None and not msg["table"].empty:
                    st.dataframe(msg["table"], use_container_width=True, hide_index=True)

        user_query = st.chat_input("Ask about startups in the directory...")
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            result = answer_query(user_query, df)
            st.session_state.last_results = result["data"] if isinstance(result.get("data"), pd.DataFrame) else pd.DataFrame()

            st.session_state.chat_history.append({"role": "assistant", "content": result["message"]})

            if isinstance(result.get("data"), pd.DataFrame) and not result["data"].empty:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": "Matched records:",
                        "table": list_output(result["data"], limit=200),
                    }
                )

            st.rerun()

    with tab_table:
        st.subheader("Directory table")
        search_text = st.text_input(
            "Filter rows",
            placeholder="Search by company, country, solution type, use case, funding stage...",
        )

        view = df.copy()
        if search_text.strip():
            q = normalize_query_terms(search_text)
            mask = view["_search_blob"].str.contains(re.escape(q), case=False, na=False)
            if "Vendor Name" in view.columns:
                mask = mask | view["Vendor Name"].astype(str).str.contains(
                    re.escape(search_text), case=False, na=False
                )
            view = view[mask]

        show_cols = [c for c in DISPLAY_COLUMNS if c in view.columns]
        st.write(f"Showing **{len(view)}** record(s).")
        st.dataframe(view[show_cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
