"""
Developmental Toxicity Prediction Web Application (Streamlit)

Local usage:
    streamlit run app.py

Required files in the same directory:
    - PubChem_RF_best.pkl                 (trained Random Forest model)
    - pubchem_preprocessor.pkl            (preprocessor from build_preprocessor.py)
    - pubchem_utils.py                    (shared utility module)
    - body.svg                            (hero icon for all pages)
    - Dev_Figure_introduction.png         (pipeline diagram for Introduction page)
"""

import base64
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from pubchem_utils import compute_raw_pubchem_fingerprint

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Developmental Toxicity Prediction",
    page_icon="👩‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: load local SVG as base64 data-URI
# ─────────────────────────────────────────────────────────────────────────────
def svg_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# Hero icon — load body.svg once globally for all pages
# ─────────────────────────────────────────────────────────────────────────────
try:
    svg_b64 = svg_to_b64("body.svg")
    HERO_ICON_HTML = (
        f'<img src="data:image/svg+xml;base64,{svg_b64}" '
        f'width="64" height="64" style="object-fit:contain;">'
    )
except FileNotFoundError:
    HERO_ICON_HTML = "👩‍⚕️"  # fallback only if body.svg is missing

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset ───────────────────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stAppViewContainer"] { background: #f5f7fa; }

.block-container {
    padding: 2.5rem 3.5rem 2rem 3.5rem !important;
    max-width: 1060px;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #dbeafe !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 !important;
}

/* Sidebar heading */
.nav-heading {
    color: #1e3a5f;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 28px 22px 16px 22px;
    border-bottom: 1px solid rgba(30,58,95,0.18);
    margin-bottom: 6px;
}

/* Hide the radio widget's own label */
section[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}
/* Stack items vertically, no gap */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
}
/* Each nav label */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    padding: 13px 22px !important;
    margin: 0 !important;
    border-radius: 0 !important;
    color: #4b6a8a !important;
    font-size: 15px !important;
    font-weight: 400 !important;
    cursor: pointer !important;
    transition: background 0.14s, color 0.14s;
    background: transparent !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
    background: rgba(66,153,225,0.15) !important;
    color: #1e3a5f !important;
}
/* Active / selected item */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input[type="radio"]:checked) {
    background: #4299e1 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
/* Hide radio circles */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label span:first-child {
    display: none !important;
}

/* ── Page title ──────────────────────────────────────────────────────────── */
.page-title {
    font-size: 36px;
    font-weight: 800;
    color: #2d3748;
    text-align: center;
    margin: 0 0 2rem 0;
    letter-spacing: -0.3px;
}

/* ── Section hero row ────────────────────────────────────────────────────── */
.hero-row {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 2rem;
}
.hero-icon img {
    width: 64px;
    height: 64px;
    object-fit: contain;
}
.hero-label {
    font-size: 22px;
    font-weight: 700;
    color: #2d3748;
    line-height: 1.3;
}

/* ── White content card ──────────────────────────────────────────────────── */
.content-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 32px 40px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    margin-bottom: 1.5rem;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 15px;
    font-weight: 500;
    color: #718096;
    padding: 12px 28px;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #4299e1 !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #4299e1 !important;
}

/* ── Result labels ───────────────────────────────────────────────────────── */
.result-positive {
    background: #fff5f5;
    border-left: 4px solid #fc8181;
    border-radius: 6px;
    padding: 14px 20px;
    font-weight: 600;
    color: #c53030;
    font-size: 15px;
}
.result-negative {
    background: #f0fff4;
    border-left: 4px solid #68d391;
    border-radius: 6px;
    padding: 14px 20px;
    font-weight: 600;
    color: #276749;
    font-size: 15px;
}

/* ── Contact card ────────────────────────────────────────────────────────── */
.contact-box {
    background: #ebf8ff;
    border-radius: 12px;
    padding: 26px 32px;
    text-align: center;
}

/* ── Footer disclaimer ───────────────────────────────────────────────────── */
.disclaimer {
    text-align: center;
    color: #a0aec0;
    font-size: 13px;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="nav-heading">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        label="nav",
        options=["Introduction", "Prediction", "Contact"],
        label_visibility="collapsed",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached across sessions)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH        = "PubChem_RF_best.pkl"
PREPROCESSOR_PATH = "pubchem_preprocessor.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    prep  = joblib.load(PREPROCESSOR_PATH)
    return model, prep["imputer"], prep["scaler"], prep["columns"]


model, imputer, scaler, feature_cols = load_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────────────────────────
def predict_smiles_list(smiles_list: list, threshold: float = 0.5) -> pd.DataFrame:
    """
    Predict developmental toxicity for a batch of SMILES strings.
    Entries that cannot be parsed by RDKit are skipped and marked accordingly.
    """
    n = len(smiles_list)
    valid_idx, valid_smiles = [], []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is not None:
            valid_idx.append(i)
            valid_smiles.append(str(smi))

    results = pd.DataFrame({
        "Prediction":  ["Invalid SMILES, skipped"] * n,
        "Probability": [np.nan] * n,
    })

    if not valid_smiles:
        return results

    raw_fp = compute_raw_pubchem_fingerprint(valid_smiles)

    if raw_fp.shape[0] != len(valid_smiles):
        raise RuntimeError(
            "PaDEL returned an inconsistent number of rows. "
            "Prediction aborted to avoid misalignment."
        )

    missing = [c for c in feature_cols if c not in raw_fp.columns]
    if missing:
        raise RuntimeError(
            f"{len(missing)} fingerprint bit(s) are missing. "
            "Please verify your PaDEL-Descriptor / Java installation."
        )

    X = raw_fp[feature_cols]
    X = pd.DataFrame(imputer.transform(X), columns=feature_cols)
    X = pd.DataFrame(scaler.transform(X),  columns=feature_cols)
    proba = model.predict_proba(X)[:, 1]

    for j, i in enumerate(valid_idx):
        p = float(proba[j])
        results.loc[i, "Probability"] = round(p, 4)
        results.loc[i, "Prediction"]  = (
            "Positive (Developmental Toxicity Risk)"
            if p >= threshold else
            "Negative (Low Risk)"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Introduction
# ─────────────────────────────────────────────────────────────────────────────
if page == "Introduction":
    st.markdown('<h1 class="page-title">Introduction</h1>', unsafe_allow_html=True)

    # Hero row — uses global HERO_ICON_HTML (body.svg)
    st.markdown(f"""
    <div class="hero-row">
        <div class="hero-icon">{HERO_ICON_HTML}</div>
        <div class="hero-label">Developmental Toxicity Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline diagram — replaced with Dev_Figure_introduction.png
    st.image("Dev_Figure_introduction.png", use_container_width=True)

    # Description card
    st.markdown("""
    <div class="content-card">
        <p style="font-size:16px;line-height:1.9;color:#4a5568;text-align:justify;margin-bottom:14px;">
        This web application is designed to predict whether a chemical compound exhibits
        <strong>developmental toxicity</strong> based on its molecular structure represented as a
        <strong>SMILES</strong> string. The prediction model employs
        <strong>PubChem molecular fingerprints (881 bits)</strong> combined with an optimized
        <strong>Random Forest classifier</strong>. Molecular fingerprints are generated using
        PaDEL-Descriptor, followed by missing-value imputation and standard scaling prior to
        classification.
        </p>
        <p style="font-size:16px;line-height:1.9;color:#4a5568;text-align:justify;margin-bottom:14px;">
        The tool supports two prediction modes:
        <strong>Single SMILES Prediction</strong> for querying individual compounds, and
        <strong>Batch Prediction</strong> via CSV file upload for high-throughput screening workflows.
        A user-adjustable classification threshold allows fine-tuning of the
        sensitivity–specificity trade-off for specific research needs.
        </p>
        <p style="font-size:13px;color:#a0aec0;margin:0;">
        ⚠️ This tool is intended for <em>research purposes only</em> and cannot replace
        in vitro / in vivo experimental assays or professional toxicological evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Prediction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Prediction":
    st.markdown('<h1 class="page-title">Prediction</h1>', unsafe_allow_html=True)

    # Hero row — uses global HERO_ICON_HTML (body.svg)
    st.markdown(f"""
    <div class="hero-row">
        <div class="hero-icon">{HERO_ICON_HTML}</div>
        <div class="hero-label">Developmental Toxicity Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    # Threshold slider
    threshold = st.slider(
        "Classification Threshold — samples with predicted probability ≥ threshold are classified as **Positive**",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["   Single SMILES Prediction   ",
                           "   Batch SMILES Prediction (CSV)   "])

    # ── Tab 1 : Single compound ───────────────────────────────────────────────
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        smiles_input = st.text_input(
            "Enter One SMILES:",
            placeholder="For example: CC(=O)OC1=CC=CC=C1C(=O)O",
        )

        if st.button("Submit", key="single_btn", type="primary"):
            if not smiles_input.strip():
                st.warning("Please enter a SMILES string before submitting.")
            else:
                with st.spinner("Computing molecular fingerprints and running the classifier…"):
                    try:
                        res = predict_smiles_list([smiles_input.strip()], threshold)
                        row = res.iloc[0]
                        st.markdown("<br>", unsafe_allow_html=True)

                        if row["Prediction"] == "Invalid SMILES, skipped":
                            st.error(
                                "The SMILES string could not be parsed by RDKit. "
                                "Please verify the structural notation."
                            )
                        else:
                            col_prob, col_result = st.columns([1, 2], gap="large")
                            with col_prob:
                                st.metric(
                                    label="Predicted Probability",
                                    value=f"{float(row['Probability']):.2%}",
                                )
                            with col_result:
                                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                                if "Positive" in row["Prediction"]:
                                    st.markdown(
                                        f'<div class="result-positive">⚠️ {row["Prediction"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        f'<div class="result-negative">✅ {row["Prediction"]}</div>',
                                        unsafe_allow_html=True,
                                    )
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

    # ── Tab 2 : Batch CSV ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Upload a CSV file containing a column named **`SMILES`**.")

        uploaded = st.file_uploader("Choose CSV File", type=["csv"])

        if uploaded is not None:
            df_in = pd.read_csv(uploaded)
            if "SMILES" not in df_in.columns:
                st.error(
                    "Column `SMILES` was not found in the uploaded file. "
                    "Please verify the column header."
                )
            else:
                st.write(f"Loaded **{len(df_in)}** molecules. Preview (first 5 rows):")
                st.dataframe(df_in.head(), use_container_width=True)

                if st.button("Run Batch Prediction", key="batch_btn", type="primary"):
                    with st.spinner(
                        f"Processing {len(df_in)} molecule(s) — "
                        "PaDEL-Descriptor computation may take a moment…"
                    ):
                        try:
                            res = predict_smiles_list(df_in["SMILES"].tolist(), threshold)
                            out = pd.concat(
                                [df_in.reset_index(drop=True), res.reset_index(drop=True)],
                                axis=1,
                            )
                            st.success(f"Prediction completed — {len(out)} results ready.")
                            st.dataframe(out, use_container_width=True)
                            csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                label="⬇ Download Prediction Results (CSV)",
                                data=csv_bytes,
                                file_name="developmental_toxicity_predictions.csv",
                                mime="text/csv",
                            )
                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Contact
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Contact":
    st.markdown('<h1 class="page-title">Contact</h1>', unsafe_allow_html=True)

    _, col_mid, _ = st.columns([1, 3, 1])
    with col_mid:
        st.markdown("""
        <div class="content-card" style="text-align:center;">
            <p style="font-size:16px;color:#4a5568;margin-bottom:28px;">
                Welcome to use this tool for free!<br>
                For any questions or feedback, please contact:
            </p>
            <div class="contact-box">
                <p style="font-size:15px;color:#4a5568;margin-bottom:10px;">
                    <strong>Contributors:</strong> Zhen Yang
                </p>
                <p style="font-size:15px;margin:0;">
                    <strong>Contact Email:</strong>
                    <a href="mailto:zhenyang_st@rcees.ac.cn"
                       style="color:#4299e1;text-decoration:none;">
                        zhenyang_st@rcees.ac.cn
                    </a>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p class="disclaimer">'
    "⚠️ This tool is for research use only and cannot replace experimental assays "
    "or professional toxicological evaluation."
    "</p>",
    unsafe_allow_html=True,
)