import streamlit as st
import pandas as pd
import torch

from bert_score import score
from transformers import AutoTokenizer

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="BERTScore Component Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 BERTScore Component Analyzer")
st.caption(
    "Compare two strings and inspect BERTScore precision, recall, "
    "F1 and token-level components."
)


# =========================================================
# CACHED COMPONENTS
# =========================================================

@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer(model_type: str):
    """
    Load tokenizer once per model.
    Cached across Streamlit reruns and user sessions.
    """
    return AutoTokenizer.from_pretrained(model_type)


@st.cache_resource(show_spinner="Loading BERTScore model...")
def load_bertscore_model(model_type: str):
    """
    Load BERTScore model once.

    bert_score internally manages the model, but explicitly
    loading it here prevents repeated Hugging Face model
    downloads/tokenizer initialization when doing additional
    token-level analysis.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_type)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    return model, device


# =========================================================
# INPUT
# =========================================================

col1, col2 = st.columns(2)

with col1:
    candidate = st.text_area(
        "Candidate / Generated Text",
        value="The cat is sitting on the mat.",
        height=150,
        key="candidate_text"
    )

with col2:
    reference = st.text_area(
        "Reference / Ground Truth Text",
        value="A cat is sitting on a mat.",
        height=150,
        key="reference_text"
    )


# =========================================================
# SETTINGS
# =========================================================

with st.expander("⚙️ BERTScore Settings", expanded=True):

    c1, c2, c3 = st.columns(3)

    with c1:
        lang = st.selectbox(
            "Language",
            ["en", "hi", "fr", "de", "es"],
            index=0
        )

    with c2:
        model_type = st.text_input(
            "Model",
            value="roberta-large"
        )

    with c3:
        use_idf = st.checkbox(
            "Use IDF weighting",
            value=False
        )


# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

st.caption(f"🖥️ Device: `{device}`")


# =========================================================
# CALCULATE
# =========================================================

if st.button(
    "🚀 Calculate BERTScore",
    type="primary",
    use_container_width=True
):

    if not candidate.strip() or not reference.strip():
        st.warning("Please enter both candidate and reference text.")
        st.stop()

    # -----------------------------------------------------
    # LOAD CACHED COMPONENTS
    # -----------------------------------------------------

    tokenizer = load_tokenizer(model_type)

    # Load model only once and keep it cached.
    #
    # This is useful if you later want to expose the
    # embedding/similarity components.
    bert_model, model_device = load_bertscore_model(model_type)

    # -----------------------------------------------------
    # BERTSCORE
    # -----------------------------------------------------

    with st.spinner("Calculating BERTScore..."):

        try:

            P, R, F1 = score(
                [candidate],
                [reference],
                lang=lang,
                model_type=model_type,
                idf=use_idf,
                verbose=False,
                rescale_with_baseline=True
            )

            precision = float(P[0])
            recall = float(R[0])
            f1 = float(F1[0])

        except Exception as e:

            st.error(
                f"Error while calculating BERTScore: {e}"
            )

            st.stop()


    # =====================================================
    # SUMMARY
    # =====================================================

    st.subheader("📊 BERTScore")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Precision",
            f"{precision:.4f}"
        )

    with m2:
        st.metric(
            "Recall",
            f"{recall:.4f}"
        )

    with m3:
        st.metric(
            "F1",
            f"{f1:.4f}"
        )


    # =====================================================
    # COMPONENT TABLE
    # =====================================================

    st.subheader("🔍 BERTScore Components")

    component_df = pd.DataFrame({
        "Component": [
            "Precision",
            "Recall",
            "F1"
        ],
        "Score": [
            precision,
            recall,
            f1
        ]
    })

    component_df["Score"] = component_df["Score"].round(6)

    st.dataframe(
        component_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # TOKENIZATION
    # =====================================================

    st.subheader("🔤 Token-Level Analysis")

    candidate_tokens = tokenizer.tokenize(candidate)
    reference_tokens = tokenizer.tokenize(reference)


    # -----------------------------------------------------
    # CANDIDATE TOKENS
    # -----------------------------------------------------

    st.markdown("### Candidate Tokens")

    candidate_df = pd.DataFrame({
        "Position": range(
            1,
            len(candidate_tokens) + 1
        ),
        "Token": candidate_tokens
    })

    st.dataframe(
        candidate_df,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # REFERENCE TOKENS
    # -----------------------------------------------------

    st.markdown("### Reference Tokens")

    reference_df = pd.DataFrame({
        "Position": range(
            1,
            len(reference_tokens) + 1
        ),
        "Token": reference_tokens
    })

    st.dataframe(
        reference_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # RAW OUTPUT
    # =====================================================

    st.subheader("🧮 Raw BERTScore Output")

    raw_df = pd.DataFrame({
        "Metric": [
            "Precision",
            "Recall",
            "F1"
        ],
        "Tensor": [
            str(P),
            str(R),
            str(F1)
        ]
    })

    st.dataframe(
        raw_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.subheader("📌 Interpretation")

    interpretation_df = pd.DataFrame({
        "Metric": [
            "Precision",
            "Recall",
            "F1"
        ],
        "Meaning": [
            "How strongly candidate tokens are semantically supported by the reference.",
            "How much of the reference meaning is captured by the candidate.",
            "Harmonic mean of Precision and Recall."
        ],
        "Score": [
            precision,
            recall,
            f1
        ]
    })

    interpretation_df["Score"] = (
        interpretation_df["Score"].round(6)
    )

    st.dataframe(
        interpretation_df,
        use_container_width=True,
        hide_index=True
    )
