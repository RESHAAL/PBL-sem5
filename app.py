import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
import re

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "data/research_papers_real.csv"
EMBEDDINGS_PATH = "models/embeddings.pkl"
FAISS_PATH = "models/faiss_index.bin"

DEFAULT_TOP_K = 10
SEMANTIC_POOL_SIZE = 100


st.set_page_config(
    page_title="ResearchLens AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# THEME
# ============================================================
# Only CSS is used here.
# The dashboard UI itself uses native Streamlit components,
# so HTML cannot accidentally appear as visible code.

st.markdown(
    """
    <style>
    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #0b1220;
        color: #e8f1ff;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #eaf4ff !important;
    }

    p, li, label {
        color: #d8e4f2 !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #111a2b !important;
        border-right: 1px solid #263956 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #e8f1ff !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #e8f1ff !important;
    }


    /* ========================================================
       INPUTS
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #111a2b !important;
        border-color: #334966 !important;
    }

    div[data-baseweb="input"] > div {
        background-color: #111a2b !important;
        border-color: #334966 !important;
    }

    input {
        color: #eef6ff !important;
        background-color: #111a2b !important;
    }

    input::placeholder {
        color: #8fa2bb !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        background-color: #ff5b61 !important;
        color: white !important;
        border: none !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        min-height: 2.7rem;
    }

    .stButton > button:hover {
        background-color: #e94b52 !important;
        color: white !important;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background-color: #111a2b;
        border: 1px solid #2a405f;
        border-radius: 13px;
        padding: 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #9eb0c8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #eaf4ff !important;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    div[data-testid="stExpander"] {
        background-color: #111a2b;
        border: 1px solid #2a405f;
        border-radius: 10px;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #263956 !important;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* ========================================================
       TEXT HELP
       ======================================================== */

    .search-help {
        color: #9eb0c8 !important;
        font-size: 0.9rem;
        margin-top: -0.3rem;
        margin-bottom: 0.7rem;
    }

    .section-kicker {
        color: #72c7f5 !important;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #8fd8ff !important;
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
    }

    .hero-description {
        color: #b7c5d8 !important;
        text-align: center;
        max-width: 950px;
        margin: 0.7rem auto 1rem auto;
        line-height: 1.6;
    }

    .sidebar-title {
        color: #eaf4ff !important;
        font-size: 1.35rem;
        font-weight: 800;
        text-align: center;
    }

    .sidebar-text {
        color: #aab8cc !important;
        font-size: 0.84rem;
        text-align: center;
        line-height: 1.5;
    }

    .card-title {
        color: #72c7f5 !important;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .big-gap {
        color: #eef6ff !important;
        font-size: 1.45rem;
        font-weight: 800;
    }

    .small-muted {
        color: #9eb0c8 !important;
        font-size: 0.85rem;
    }

    .accent-text {
        color: #72c7f5 !important;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():
    data = pd.read_csv(DATASET_PATH)

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].fillna("")

    return data


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_embeddings():
    with open(EMBEDDINGS_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_faiss():
    return faiss.read_index(FAISS_PATH)


try:
    df = load_dataset()
    embedding_data = load_embeddings()
    index = load_faiss()
    model = load_model()

except Exception as e:
    st.error(
        "The research data or AI model could not be loaded."
    )
    st.code(str(e))
    st.stop()


# ============================================================
# DOMAIN LIST
# ============================================================

DOMAIN_ORDER = [
    "Agriculture",
    "Computer Science",
    "Economics and Finance",
    "Education",
    "Energy",
    "Engineering",
    "Environmental Science",
    "Healthcare",
    "Materials Science",
    "Social Sciences"
]

available_domains = [
    domain
    for domain in DOMAIN_ORDER
    if domain in df["Domain"].astype(str).unique()
]

DOMAIN_OPTIONS = available_domains + ["All Domains"]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🔎 ResearchLens AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-text">'
        'Explore research, identify meaningful gaps, '
        'and discover future research directions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="card-title">Research Domain</div>',
        unsafe_allow_html=True
    )

    domain = st.selectbox(
        "Research Domain",
        DOMAIN_OPTIONS,
        index=(
            DOMAIN_OPTIONS.index("Healthcare")
            if "Healthcare" in DOMAIN_OPTIONS
            else 0
        ),
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="card-title">Relevant Papers</div>',
        unsafe_allow_html=True
    )

    top_k = st.slider(
        "Number of papers",
        min_value=5,
        max_value=20,
        value=DEFAULT_TOP_K
    )

    st.divider()

    st.markdown(
        '<div class="card-title">Dataset Coverage</div>',
        unsafe_allow_html=True
    )

    st.write("📚 **2,200+** research papers")
    st.write("🌐 **10** research domains")
    st.write("🧾 **1,900+** unique papers")

    st.divider()

    st.caption(
        "Semantic search uses Sentence-BERT embeddings "
        "and FAISS similarity search."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="
            color:#eaf4ff;
            font-size:3rem;
            margin-bottom:0.3rem;
        ">
            🔎 ResearchLens AI
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'AI-Driven Semantic Analysis for Automated Research Gap Identification'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-description">'
    'Enter a research topic and discover relevant papers, '
    'the main research gap, SDG alignment, and future research directions.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# STATISTICS
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Research Papers",
        "2,200+"
    )

with c2:
    st.metric(
        "Research Domains",
        "10"
    )

with c3:
    st.metric(
        "Unique Papers",
        "1,900+"
    )


st.divider()


# ============================================================
# SEARCH SECTION
# ============================================================

st.markdown(
    '<div class="section-kicker">Research Discovery</div>',
    unsafe_allow_html=True
)

st.subheader("What is the research gap in your topic?")

st.markdown(
    '<div class="search-help">'
    'Enter a research topic to find semantically relevant literature '
    'and identify the main gap in that topic.'
    '</div>',
    unsafe_allow_html=True
)

topic = st.text_input(
    "Research Topic",
    placeholder="Example: AI for disease detection",
    label_visibility="collapsed"
)

search_button = st.button(
    "🔎 Analyze Research Landscape",
    use_container_width=True
)


# ============================================================
# GAP PATTERNS
# ============================================================

GAP_PATTERNS = {

    "Limited Dataset": [
        "small dataset",
        "limited dataset",
        "insufficient dataset",
        "limited data",
        "lack of data",
        "data scarcity",
        "data limitation",
        "limited training data"
    ],

    "Limited Sample Size": [
        "small sample",
        "limited sample size",
        "small sample size",
        "limited number of participants"
    ],

    "Limited Generalizability": [
        "limited generalizability",
        "poor generalizability",
        "lack of generalizability",
        "limited generalisation",
        "poor generalisation",
        "generalizability remains"
    ],

    "Limited External Validation": [
        "lack of external validation",
        "limited external validation",
        "no external validation",
        "without external validation",
        "external validation is limited",
        "independent validation is limited"
    ],

    "Limited Real-World Validation": [
        "limited real-world",
        "lack of real-world",
        "real-world deployment",
        "real time deployment",
        "real-time deployment",
        "not deployed",
        "clinical deployment",
        "real world validation",
        "real-world validation",
        "field validation",
        "practical deployment"
    ],

    "Poor Scalability": [
        "poor scalability",
        "limited scalability",
        "lack of scalability",
        "scalability remains",
        "scalability challenge"
    ],

    "Limited Interpretability": [
        "lack of interpretability",
        "limited interpretability",
        "poor interpretability",
        "lack of explainability",
        "limited explainability",
        "black box"
    ],

    "Limited Longitudinal Studies": [
        "limited longitudinal",
        "lack of longitudinal",
        "few longitudinal",
        "longitudinal studies are limited",
        "short-term study",
        "short term study"
    ],

    "Limited Diversity": [
        "limited diversity",
        "insufficient diversity",
        "lack of diversity",
        "limited demographic diversity",
        "limited geographic diversity",
        "underrepresented population",
        "underrepresented populations"
    ],

    "Resource Constraints": [
        "resource-constrained",
        "resource constrained",
        "limited computational resources",
        "low-resource",
        "resource limitations",
        "computationally expensive"
    ],

    "Limited Comparative Evaluation": [
        "limited comparative",
        "lack of comparative",
        "few comparative studies",
        "limited comparison",
        "lack of comparison"
    ],

    "Limited Clinical Validation": [
        "limited clinical validation",
        "lack of clinical validation",
        "clinical validation is limited",
        "limited validation",
        "clinical evidence is limited",
        "clinical testing is limited"
    ],

    "Ethical Concerns": [
        "ethical concerns",
        "ethical challenges",
        "lack of ethical",
        "ethical issues",
        "privacy concerns",
        "privacy challenges",
        "fairness concerns",
        "bias concerns",
        "algorithmic bias"
    ]
}


# ============================================================
# GAP DETECTION
# ============================================================

def detect_gaps_from_text(text):

    if not text:
        return []

    text = str(text).lower()

    found = []

    for gap_name, patterns in GAP_PATTERNS.items():

        for pattern in patterns:

            if re.search(
                re.escape(pattern),
                text
            ):
                found.append(gap_name)
                break

    return found


def normalize_existing_gap(value):

    if not value:
        return []

    text = str(value)

    if text.lower() == "nan":
        return []

    aliases = {
        "Limited real-world deployment":
            "Limited Real-World Validation",

        "Limited clinical validation":
            "Limited Clinical Validation",

        "Limited comparative evaluation":
            "Limited Comparative Evaluation",

        "Limited interpretability":
            "Limited Interpretability",

        "Limited dataset":
            "Limited Dataset",

        "Limited sample size":
            "Limited Sample Size",

        "Limited external validation":
            "Limited External Validation",

        "Limited generalizability":
            "Limited Generalizability",

        "Limited diversity":
            "Limited Diversity",

        "Poor scalability":
            "Poor Scalability",

        "Resource constraints":
            "Resource Constraints",

        "Ethical concerns":
            "Ethical Concerns"
    }

    found = []

    for item in text.split(";"):

        item = item.strip()

        if not item:
            continue

        for canonical in GAP_PATTERNS.keys():

            if item.lower() == canonical.lower():
                found.append(canonical)
                break

        for old, new in aliases.items():

            if item.lower() == old.lower():
                found.append(new)
                break

    return list(dict.fromkeys(found))


def get_paper_gap_evidence(paper):

    gaps = []

    # --------------------------------------------------------
    # 1. Use existing Detected_Gaps from dataset
    # --------------------------------------------------------

    if "Detected_Gaps" in paper.index:

        gaps.extend(
            normalize_existing_gap(
                paper.get(
                    "Detected_Gaps",
                    ""
                )
            )
        )

    # --------------------------------------------------------
    # 2. Enrich using paper text
    # --------------------------------------------------------

    text = " ".join(
        [
            str(paper.get("Title", "")),
            str(paper.get("Abstract", "")),
            str(paper.get("Keywords", "")),
            str(paper.get("Topics", ""))
        ]
    )

    gaps.extend(
        detect_gaps_from_text(text)
    )

    return list(dict.fromkeys(gaps))


def calculate_gap_strength(
    count,
    relevant_count
):

    if relevant_count <= 0:
        return "Low"

    percentage = (
        count / relevant_count
    ) * 100

    if percentage >= 30:
        return "High"

    if percentage >= 15:
        return "Medium"

    return "Low"


def gap_explanation(gap):

    explanations = {

        "Limited Dataset":
            "Many studies rely on small or limited datasets, "
            "which can reduce reliability and make results harder "
            "to generalize.",

        "Limited Sample Size":
            "A small number of participants or observations can "
            "make findings less reliable and may not represent "
            "the wider population.",

        "Limited Generalizability":
            "Studies are often tested in specific settings, so "
            "it remains unclear whether the results will work "
            "equally well in other datasets or populations.",

        "Limited External Validation":
            "Many studies are evaluated on internal data, while "
            "testing on independent datasets remains limited.",

        "Limited Real-World Validation":
            "Research often focuses on experiments and model "
            "performance, while real-world testing and practical "
            "deployment are less explored.",

        "Poor Scalability":
            "Some approaches work in controlled experiments but "
            "may become difficult to use when data, users or "
            "applications increase.",

        "Limited Interpretability":
            "Complex AI models can produce accurate results "
            "without clearly explaining why a prediction was made.",

        "Limited Longitudinal Studies":
            "Many studies use short observation periods, leaving "
            "long-term performance and impact less understood.",

        "Limited Diversity":
            "Limited demographic, geographic or environmental "
            "diversity can reduce fairness and generalizability.",

        "Resource Constraints":
            "Some methods require high computational power or "
            "expensive resources, limiting practical adoption.",

        "Limited Comparative Evaluation":
            "There are not enough direct comparisons between "
            "different methods, making it difficult to identify "
            "the most suitable approach.",

        "Limited Clinical Validation":
            "A method may show promising experimental results "
            "but still lack enough validation in real clinical settings.",

        "Ethical Concerns":
            "Privacy, fairness, transparency, bias and responsible "
            "use remain important challenges when applying AI "
            "in real-world settings."
    }

    return explanations.get(
        gap,
        "The literature shows an area that needs further investigation."
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

RECOMMENDATIONS = {

    "Limited Dataset": {
        "direction":
            "Build larger and more diverse datasets that represent "
            "different real-world conditions.",

        "actions": [
            "Collect data from multiple reliable sources.",
            "Increase dataset size and diversity.",
            "Use data augmentation where appropriate.",
            "Validate results on unseen data."
        ],

        "methodology":
            "Multi-source data collection, data augmentation "
            "and cross-dataset validation.",

        "contribution":
            "More robust models with better generalization."
    },

    "Limited Sample Size": {
        "direction":
            "Increase the study sample using data from multiple "
            "populations or sources.",

        "actions": [
            "Use a larger sample size.",
            "Include different populations or conditions.",
            "Perform statistical power analysis.",
            "Validate results on an independent sample."
        ],

        "methodology":
            "Multi-source sampling, statistical validation "
            "and independent testing.",

        "contribution":
            "More reliable and representative research findings."
    },

    "Limited Generalizability": {
        "direction":
            "Test the proposed approach across different datasets, "
            "populations and environments.",

        "actions": [
            "Evaluate on unseen datasets.",
            "Test across different populations.",
            "Compare performance across environments.",
            "Report generalization performance."
        ],

        "methodology":
            "Cross-dataset and cross-environment evaluation.",

        "contribution":
            "More reliable real-world performance."
    },

    "Limited External Validation": {
        "direction":
            "Validate existing approaches using independent external datasets.",

        "actions": [
            "Use an unseen external dataset.",
            "Test across different data sources.",
            "Compare internal and external performance.",
            "Report robustness and generalization."
        ],

        "methodology":
            "External validation, cross-dataset testing "
            "and robustness analysis.",

        "contribution":
            "Stronger evidence that the approach works beyond "
            "the original dataset."
    },

    "Limited Real-World Validation": {
        "direction":
            "Move beyond experimental results and validate the "
            "approach in real-world environments.",

        "actions": [
            "Build a working prototype.",
            "Conduct field or real-world testing.",
            "Measure performance under practical conditions.",
            "Study deployment challenges and user feedback."
        ],

        "methodology":
            "Prototype deployment, field testing and real-world "
            "performance evaluation.",

        "contribution":
            "Better practical usability and deployment readiness."
    },

    "Poor Scalability": {
        "direction":
            "Design scalable approaches that can handle larger "
            "datasets, users and real-world workloads.",

        "actions": [
            "Test with larger datasets.",
            "Measure computational cost.",
            "Optimize model performance.",
            "Evaluate cloud or edge deployment."
        ],

        "methodology":
            "Scalability testing, model optimization "
            "and deployment evaluation.",

        "contribution":
            "More efficient and scalable systems."
    },

    "Limited Interpretability": {
        "direction":
            "Improve transparency so researchers and users "
            "can understand model predictions.",

        "actions": [
            "Explain important model predictions.",
            "Identify influential input features.",
            "Compare explanation methods.",
            "Evaluate explanation consistency."
        ],

        "methodology":
            "Explainable AI, SHAP, LIME and feature-attribution analysis.",

        "contribution":
            "More transparent and trustworthy AI systems."
    },

    "Limited Longitudinal Studies": {
        "direction":
            "Study the long-term performance and impact "
            "of the proposed approach.",

        "actions": [
            "Use longer observation periods.",
            "Track performance over time.",
            "Study changes in real-world conditions.",
            "Compare short-term and long-term results."
        ],

        "methodology":
            "Longitudinal study design and repeated performance evaluation.",

        "contribution":
            "Better understanding of long-term effectiveness."
    },

    "Limited Diversity": {
        "direction":
            "Increase demographic, geographic and environmental "
            "diversity in the research data.",

        "actions": [
            "Include underrepresented groups.",
            "Collect data from different locations.",
            "Analyze possible dataset bias.",
            "Compare performance across groups."
        ],

        "methodology":
            "Diverse data collection, subgroup analysis "
            "and bias evaluation.",

        "contribution":
            "Fairer and more generalizable research outcomes."
    },

    "Resource Constraints": {
        "direction":
            "Develop efficient methods that can operate under "
            "realistic computational and infrastructure limits.",

        "actions": [
            "Reduce model complexity.",
            "Evaluate computational requirements.",
            "Use model compression where appropriate.",
            "Explore edge or low-resource deployment."
        ],

        "methodology":
            "Model optimization, quantization and resource-aware evaluation.",

        "contribution":
            "More efficient and accessible AI systems."
    },

    "Limited Comparative Evaluation": {
        "direction":
            "Perform systematic comparisons between existing "
            "approaches using the same evaluation setting.",

        "actions": [
            "Select strong baseline methods.",
            "Use the same dataset and metrics.",
            "Compare strengths and weaknesses.",
            "Report statistical differences."
        ],

        "methodology":
            "Benchmarking, comparative experiments "
            "and statistical evaluation.",

        "contribution":
            "Clearer evidence about which methods perform best."
    },

    "Limited Clinical Validation": {
        "direction":
            "Validate promising AI methods using real clinical "
            "data and realistic clinical workflows.",

        "actions": [
            "Use independent clinical datasets.",
            "Test with realistic clinical workflows.",
            "Measure safety and reliability.",
            "Evaluate performance across patient groups."
        ],

        "methodology":
            "Clinical validation, external testing "
            "and workflow-based evaluation.",

        "contribution":
            "Stronger evidence for safe and practical clinical use."
    },

    "Ethical Concerns": {
        "direction":
            "Develop transparent, fair and responsible AI systems "
            "that can be trusted in real-world use.",

        "actions": [
            "Evaluate privacy risks.",
            "Measure fairness and possible bias.",
            "Improve model transparency.",
            "Define responsible deployment requirements."
        ],

        "methodology":
            "Fairness evaluation, privacy-preserving AI, "
            "bias analysis and Explainable AI.",

        "contribution":
            "More trustworthy and responsible AI adoption."
    }
}


# ============================================================
# SDG MAP
# ============================================================

SDG_MAP = {

    "SDG 1": "No Poverty",
    "SDG 2": "Zero Hunger",
    "SDG 3": "Good Health and Well-Being",
    "SDG 4": "Quality Education",
    "SDG 5": "Gender Equality",
    "SDG 6": "Clean Water and Sanitation",
    "SDG 7": "Affordable and Clean Energy",
    "SDG 8": "Decent Work and Economic Growth",
    "SDG 9": "Industry, Innovation and Infrastructure",
    "SDG 10": "Reduced Inequalities",
    "SDG 11": "Sustainable Cities and Communities",
    "SDG 12": "Responsible Consumption and Production",
    "SDG 13": "Climate Action",
    "SDG 14": "Life Below Water",
    "SDG 15": "Life on Land",
    "SDG 16": "Peace, Justice and Strong Institutions",
    "SDG 17": "Partnerships for the Goals"
}


def normalize_sdg(value):

    if not value:
        return None

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    match = re.search(
        r"SDG\s*(\d{1,2})",
        text,
        flags=re.IGNORECASE
    )

    if match:

        number = int(match.group(1))
        key = f"SDG {number}"

        if key in SDG_MAP:

            return {
                "number": key,
                "name": SDG_MAP[key]
            }

    lower = text.lower()

    for key, name in SDG_MAP.items():

        if name.lower() in lower:

            return {
                "number": key,
                "name": name
            }

    return None


def sdg_relevance(
    sdg_number,
    topic
):

    relevance = {

        "SDG 1":
            "The research may support better economic access, "
            "livelihoods or resource allocation.",

        "SDG 2":
            "The research can support food production, agricultural "
            "productivity, food security or sustainable farming.",

        "SDG 3":
            "The research is relevant to health, disease prevention, "
            "diagnosis, treatment or healthcare delivery.",

        "SDG 4":
            "The research can support learning, teaching, educational "
            "access or better education systems.",

        "SDG 5":
            "The research can contribute to gender-inclusive systems "
            "and opportunities.",

        "SDG 6":
            "The research can support efficient water use, water "
            "quality or sustainable water management.",

        "SDG 7":
            "The research can support cleaner energy, energy "
            "efficiency or reliable energy systems.",

        "SDG 8":
            "The research can support productivity, employment, "
            "economic activity or efficient work.",

        "SDG 9":
            "The research is connected to innovation, technology "
            "development, infrastructure or advanced AI systems.",

        "SDG 10":
            "The research can help reduce inequalities through "
            "more inclusive and accessible systems.",

        "SDG 11":
            "The research can support safer, smarter and more "
            "sustainable communities and urban systems.",

        "SDG 12":
            "The research can support efficient resource use, "
            "sustainable production and reduced waste.",

        "SDG 13":
            "The research can contribute to climate monitoring, "
            "mitigation, adaptation or environmental sustainability.",

        "SDG 14":
            "The research can support protection and sustainable "
            "use of marine and aquatic ecosystems.",

        "SDG 15":
            "The research can support biodiversity, ecosystems, "
            "forests, land management or conservation.",

        "SDG 16":
            "The research can support responsible institutions, "
            "governance, transparency or trustworthy systems.",

        "SDG 17":
            "The research can support collaboration, knowledge "
            "sharing and partnerships."
    }

    return relevance.get(
        sdg_number,
        f"This SDG has a measurable connection with research related to {topic}."
    )


# ============================================================
# SEARCH
# ============================================================

if search_button:

    if not topic.strip():

        st.warning(
            "Please enter a research topic first."
        )

        st.stop()


    # ========================================================
    # DOMAIN FILTER
    # ========================================================

    if domain == "All Domains":

        filtered_df = df.copy()

    else:

        filtered_df = df[
            df["Domain"].astype(str).str.strip()
            == domain
        ].copy()


    if filtered_df.empty:

        st.warning(
            "No papers are available for the selected domain."
        )

        st.stop()


    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    query_embedding = model.encode(
        [topic],
        normalize_embeddings=True
    ).astype("float32")


    # ========================================================
    # FAISS SEARCH
    # ========================================================

    search_k = min(
        max(
            top_k * 10,
            SEMANTIC_POOL_SIZE
        ),
        index.ntotal
    )

    distances, indices = index.search(
        query_embedding,
        search_k
    )


    # ========================================================
    # FAST PAPER LOOKUP
    # ========================================================

    paper_id_to_row = {}

    for _, row in df.iterrows():

        paper_id = str(
            row.get(
                "Paper_ID",
                ""
            )
        ).strip()

        if paper_id:

            paper_id_to_row[paper_id] = row


    # ========================================================
    # BUILD SEMANTICALLY RELEVANT POOL
    # ========================================================

    relevant_results = []

    used_ids = set()


    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(
            embedding_data["paper_ids"]
        ):
            continue


        paper_id = str(
            embedding_data["paper_ids"][idx]
        ).strip()


        if paper_id in used_ids:
            continue

        if paper_id not in paper_id_to_row:
            continue


        paper = paper_id_to_row[paper_id]


        paper_domain = str(
            paper.get(
                "Domain",
                ""
            )
        ).strip()


        if domain != "All Domains":

            if paper_domain != domain:
                continue


        used_ids.add(paper_id)


        paper_copy = paper.copy()


        # Since embeddings are normalized,
        # FAISS inner product is cosine similarity.
        similarity = float(distance) * 100

        similarity = max(
            0,
            min(
                100,
                similarity
            )
        )


        paper_copy["Similarity"] = round(
            similarity,
            2
        )


        relevant_results.append(
            paper_copy
        )


        if len(relevant_results) >= SEMANTIC_POOL_SIZE:
            break


    if not relevant_results:

        st.warning(
            "No relevant papers were found for this topic and domain."
        )

        st.stop()


    results_df = pd.DataFrame(
        relevant_results[:top_k]
    )

    relevant_pool_df = pd.DataFrame(
        relevant_results
    )


    # ========================================================
    # RESEARCH LANDSCAPE
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">Research Landscape</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        f'Literature related to "{topic.strip()}"'
    )


    landscape_col1, landscape_col2, landscape_col3 = st.columns(3)


    with landscape_col1:

        st.metric(
            "Relevant Papers",
            len(relevant_pool_df)
        )


    with landscape_col2:

        years = pd.to_numeric(
            relevant_pool_df["Year"],
            errors="coerce"
        ).dropna()

        if len(years):

            year_text = (
                f"{int(years.min())}"
                f"–"
                f"{int(years.max())}"
            )

        else:

            year_text = "N/A"


        st.metric(
            "Research Period",
            year_text
        )


    with landscape_col3:

        domain_count = relevant_pool_df[
            "Domain"
        ].nunique()

        st.metric(
            "Domains Represented",
            int(domain_count)
        )


    # ========================================================
    # RESEARCH TRENDS
    # ========================================================

    st.subheader("Research Trends")


    trend_col1, trend_col2 = st.columns(2)


    with trend_col1:

        st.caption("Papers by Year")

        year_counts = (
            pd.to_numeric(
                relevant_pool_df["Year"],
                errors="coerce"
            )
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )


        if not year_counts.empty:

            st.bar_chart(
                year_counts,
                height=240
            )

        else:

            st.info(
                "Year information is not available."
            )


    with trend_col2:

        st.caption("Most Common Research Topics")

        topic_values = []


        for value in relevant_pool_df.get(
            "Topics",
            []
        ):

            if not value:
                continue

            for item in str(value).split(";"):

                item = item.strip()

                if item:
                    topic_values.append(item)


        if topic_values:

            topic_counts = (
                pd.Series(topic_values)
                .value_counts()
                .head(8)
            )

            st.bar_chart(
                topic_counts,
                height=240
            )

        else:

            st.info(
                "Topic information is not available."
            )


    # ========================================================
    # TOP PAPERS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">Semantic Research Discovery</div>',
        unsafe_allow_html=True
    )

    st.subheader("Top Relevant Research Papers")


    for i, (_, paper) in enumerate(
        results_df.iterrows(),
        start=1
    ):

        title = str(
            paper.get(
                "Title",
                "Untitled Paper"
            )
        )

        paper_id = str(
            paper.get(
                "Paper_ID",
                "N/A"
            )
        )

        paper_domain = str(
            paper.get(
                "Domain",
                "N/A"
            )
        )

        year = str(
            paper.get(
                "Year",
                "N/A"
            )
        )

        similarity = paper.get(
            "Similarity",
            0
        )


        # ----------------------------------------------------
        # Native Streamlit card
        # ----------------------------------------------------

        with st.container(
            border=True
        ):

            st.caption(
                f"PAPER {i}"
            )

            st.markdown(
                f"### {title}"
            )

            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)


            with meta_col1:

                st.caption("Paper ID")

                st.write(paper_id)


            with meta_col2:

                st.caption("Domain")

                st.write(paper_domain)


            with meta_col3:

                st.caption("Year")

                st.write(year)


            with meta_col4:

                st.caption("Semantic Similarity")

                st.write(
                    f"{similarity}%"
                )


            abstract = str(
                paper.get(
                    "Abstract",
                    ""
                )
            ).strip()


            if abstract and abstract.lower() != "nan":

                with st.expander(
                    "📄 View Full Abstract"
                ):

                    st.write(
                        abstract
                    )


            topics = str(
                paper.get(
                    "Topics",
                    ""
                )
            ).strip()


            keywords = str(
                paper.get(
                    "Keywords",
                    ""
                )
            ).strip()


            if topics:

                st.caption(
                    f"Topics: {topics}"
                )


            if keywords:

                st.caption(
                    f"Keywords: {keywords}"
                )


    # ========================================================
    # RESEARCH GAP ANALYSIS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">Core Analysis</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "🎯 Research Gap Analysis"
    )

    st.caption(
        "The gap is identified from papers that are semantically "
        "relevant to your selected research topic."
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Only semantic relevant papers are analyzed.
    # NOT all papers from the selected domain.
    # --------------------------------------------------------

    gap_counter = {}
    gap_papers = {}


    for _, paper in relevant_pool_df.iterrows():

        detected = get_paper_gap_evidence(
            paper
        )


        for gap in detected:

            gap_counter[gap] = (
                gap_counter.get(
                    gap,
                    0
                ) + 1
            )


            gap_papers.setdefault(
                gap,
                []
            ).append(
                paper.get(
                    "Paper_ID",
                    ""
                )
            )


    primary_gap = None
    primary_count = 0
    strength = "Low"


    if gap_counter:

        gap_counts = (
            pd.Series(gap_counter)
            .sort_values(
                ascending=False
            )
        )


        primary_gap = gap_counts.index[0]

        primary_count = int(
            gap_counts.iloc[0]
        )


        strength = calculate_gap_strength(
            primary_count,
            len(relevant_pool_df)
        )


        explanation = gap_explanation(
            primary_gap
        )


        # ====================================================
        # PRIMARY GAP
        # ====================================================

        gap_col1, gap_col2, gap_col3 = st.columns(3)


        with gap_col1:

            with st.container(
                border=True
            ):

                st.caption(
                    "PRIMARY GAP"
                )

                st.markdown(
                    f"### {primary_gap}"
                )

                st.caption(
                    "Most frequently detected gap"
                )


        with gap_col2:

            with st.container(
                border=True
            ):

                st.caption(
                    "EVIDENCE"
                )

                st.metric(
                    "Relevant Papers",
                    primary_count
                )

                st.caption(
                    f"out of {len(relevant_pool_df)} relevant papers"
                )


        with gap_col3:

            with st.container(
                border=True
            ):

                st.caption(
                    "GAP STRENGTH"
                )

                if strength == "High":

                    st.error(
                        "High"
                    )

                elif strength == "Medium":

                    st.warning(
                        "Medium"
                    )

                else:

                    st.info(
                        "Low"
                    )


        # ====================================================
        # WHY THIS IS A GAP
        # ====================================================

        why_col1, why_col2 = st.columns(
            [1.2, 1]
        )


        with why_col1:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Why is this a gap?"
                )

                st.write(
                    explanation
                )

                st.write(
                    f"Evidence appears in "
                    f"**{primary_count} of "
                    f"{len(relevant_pool_df)}** "
                    f"relevant papers."
                )

                st.write(
                    "This suggests that the topic is being "
                    "studied, but this particular issue is not "
                    "fully addressed in the relevant literature."
                )


        with why_col2:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Other Gaps Found"
                )

                for gap, count in gap_counts.head(6).items():

                    st.write(
                        f"**{gap}** — {int(count)} papers"
                    )


    else:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Research Gap"
            )

            st.info(
                "No strong predefined gap pattern was found "
                "in the relevant papers."
            )

            st.write(
                "The current literature does not contain enough "
                "explicit evidence for one of the predefined "
                "gap categories. A deeper literature review "
                "is recommended before claiming a specific gap."
            )


    # ========================================================
    # SDG ALIGNMENT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">Sustainable Development</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "🌍 SDG Alignment"
    )

    st.caption(
        "SDG alignment is calculated from SDG information "
        "attached to the papers relevant to your topic."
    )


    sdg_counter = {}


    for _, paper in relevant_pool_df.iterrows():

        value = paper.get(
            "SDGs",
            ""
        )


        if not value:
            continue


        for raw_sdg in str(value).split(";"):

            parsed = normalize_sdg(
                raw_sdg
            )


            if parsed:

                key = parsed["number"]

                sdg_counter[key] = (
                    sdg_counter.get(
                        key,
                        0
                    ) + 1
                )


    if sdg_counter:

        sorted_sdgs = sorted(
            sdg_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )


        top_sdg_number = sorted_sdgs[0][0]

        top_sdg_count = sorted_sdgs[0][1]

        top_sdg_name = SDG_MAP[
            top_sdg_number
        ]


        relevance = sdg_relevance(
            top_sdg_number,
            topic
        )


        sdg_col1, sdg_col2 = st.columns(
            [1, 1.6]
        )


        with sdg_col1:

            with st.container(
                border=True
            ):

                st.caption(
                    "MOST RELEVANT SDG"
                )

                st.markdown(
                    f"### {top_sdg_number}"
                )

                st.markdown(
                    f"**{top_sdg_name}**"
                )

                st.caption(
                    f"Evidence: {top_sdg_count} relevant papers"
                )


        with sdg_col2:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Topic Connection"
                )

                st.write(
                    relevance
                )

                st.caption(
                    f"This SDG is supported by papers most relevant "
                    f"to \"{topic}\"."
                )


        if len(sorted_sdgs) > 1:

            st.markdown(
                "### Other SDGs in the Relevant Literature"
            )


            other_sdgs = sorted_sdgs[1:7]

            sdg_cols = st.columns(
                min(
                    3,
                    len(other_sdgs)
                )
            )


            for i, (
                sdg_number,
                count
            ) in enumerate(other_sdgs):

                with sdg_cols[
                    i % len(sdg_cols)
                ]:

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"**{sdg_number}**"
                        )

                        st.write(
                            SDG_MAP[
                                sdg_number
                            ]
                        )

                        st.caption(
                            f"{count} relevant papers"
                        )


    else:

        with st.container(
            border=True
        ):

            st.markdown(
                "### SDG Alignment"
            )

            st.info(
                "No strong SDG alignment was found."
            )

            st.write(
                "The relevant papers do not contain enough "
                "reliable SDG information to assign a specific SDG."
            )


    # ========================================================
    # FUTURE RESEARCH RECOMMENDATION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">Next Research Opportunity</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "💡 Future Research Recommendation"
    )


    if (
        primary_gap
        and primary_gap in RECOMMENDATIONS
    ):

        rec = RECOMMENDATIONS[
            primary_gap
        ]


        # ----------------------------------------------------
        # Recommendation direction
        # ----------------------------------------------------

        with st.container(
            border=True
        ):

            st.caption(
                "RECOMMENDED RESEARCH DIRECTION"
            )

            st.markdown(
                f"### {rec['direction']}"
            )

            st.caption(
                f"This recommendation is based on the dominant "
                f"gap detected for \"{topic}\"."
            )


        rec_col1, rec_col2 = st.columns(2)


        with rec_col1:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Suggested Research Actions"
                )

                for action in rec["actions"]:

                    st.write(
                        f"• {action}"
                    )


        with rec_col2:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Suggested Methodology"
                )

                st.write(
                    rec["methodology"]
                )

                st.markdown(
                    "### Expected Contribution"
                )

                st.write(
                    rec["contribution"]
                )


    else:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Research Recommendation"
            )

            st.info(
                "Conduct a deeper systematic review before "
                "defining a specific future research direction."
            )

            st.write(
                "The current relevant literature does not provide "
                "enough explicit evidence for one dominant "
                "predefined research gap."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ResearchLens AI • AI-Driven Semantic Analysis for "
    "Automated Research Gap Identification"
)