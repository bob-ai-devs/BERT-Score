import streamlit as st
import pandas as pd

from bert_score import score as bert_score
from transformers import AutoTokenizer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BERTScore Component Analyzer",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🧠 BERTScore Component Analyzer")

st.caption(
    "Compare a generated response with a reference and "
    "inspect Precision, Recall and F1."
)


# =========================================================
# CACHED TOKENIZER
# =========================================================

@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer(model_type: str):

    return AutoTokenizer.from_pretrained(
        model_type
    )


# =========================================================
# INPUT TEXT
# =========================================================

col1, col2 = st.columns(2)

with col1:

    our_resp = st.text_area(
        "Our Response",
        value="The cat is sitting on the mat.",
        height=180,
        key="our_response"
    )


with col2:

    reference = st.text_area(
        "Reference",
        value="A cat is sitting on a mat.",
        height=180,
        key="reference"
    )


# =========================================================
# SETTINGS
# =========================================================

with st.expander("⚙️ BERTScore Settings", expanded=True):

    c1, c2 = st.columns(2)

    # -----------------------------------------------------
    # LANGUAGE
    # -----------------------------------------------------

    with c1:

        lang = st.selectbox(
            "Language",
            options=[
                "en",
                "hi",
                "fr",
                "de",
                "es"
            ],
            index=0,
            key="bertscore_language"
        )


    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    with c2:

        model_options = {
            "Default BERTScore Model": None,
            "RoBERTa Large": "roberta-large",
            "RoBERTa Base": "roberta-base",
            "DistilRoBERTa Base": "distilroberta-base",
        }

        model_label = st.selectbox(
            "BERTScore Model",
            options=list(model_options.keys()),
            index=0,
            key="bertscore_model"
        )

        selected_model = model_options[model_label]


# =========================================================
# CALCULATE
# =========================================================

if st.button(
    "🚀 Calculate BERTScore",
    type="primary",
    use_container_width=True
):

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if not our_resp.strip():

        st.warning(
            "Please enter Our Response."
        )

        st.stop()


    if not reference.strip():

        st.warning(
            "Please enter the Reference."
        )

        st.stop()


    # =====================================================
    # BERTSCORE
    # =====================================================

    with st.spinner("Calculating BERTScore..."):

        try:

            # -------------------------------------------------
            # IMPORTANT:
            #
            # When "Default BERTScore Model" is selected,
            # this is intentionally the SAME call as your
            # previous implementation:
            #
            # P_o, R_o, F1_o = bert_score(
            #     [our_resp],
            #     [reference],
            #     lang='en',
            #     verbose=False
            # )
            # -------------------------------------------------

            if selected_model is None:

                P_o, R_o, F1_o = bert_score(
                    [our_resp],
                    [reference],
                    lang=lang,
                    verbose=False
                )

            else:

                P_o, R_o, F1_o = bert_score(
                    [our_resp],
                    [reference],
                    lang=lang,
                    model_type=selected_model,
                    verbose=False
                )


            # -------------------------------------------------
            # SAME AS YOUR EXISTING CODE
            # -------------------------------------------------

            precision = P_o.item()
            recall = R_o.item()
            f1 = F1_o.item()


        except Exception as e:

            st.error(
                f"BERTScore calculation failed: {e}"
            )

            st.stop()


    # =====================================================
    # MAIN SCORES
    # =====================================================

    st.subheader("📊 BERTScore")

    m1, m2, m3 = st.columns(3)


    with m1:

        st.metric(
            "Precision",
            f"{precision:.6f}"
        )


    with m2:

        st.metric(
            "Recall",
            f"{recall:.6f}"
        )


    with m3:

        st.metric(
            "F1",
            f"{f1:.6f}"
        )


    # =====================================================
    # COMPONENT TABLE
    # =====================================================

    st.subheader("🔍 Score Components")

    component_df = pd.DataFrame(
        {
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
        }
    )

    component_df["Score"] = component_df[
        "Score"
    ].round(6)


    st.dataframe(
        component_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # TOKENIZATION
    # =====================================================

    st.subheader("🔤 Tokenization")

    # -----------------------------------------------------
    # Determine tokenizer
    # -----------------------------------------------------

    if selected_model is not None:

        tokenizer_model = selected_model

    else:

        # BERTScore's default English model is used for
        # the normal call. For displaying tokens, use the
        # standard RoBERTa tokenizer.
        tokenizer_model = "roberta-large"


    try:

        tokenizer = load_tokenizer(
            tokenizer_model
        )

        our_tokens = tokenizer.tokenize(
            our_resp
        )

        reference_tokens = tokenizer.tokenize(
            reference
        )


        # -------------------------------------------------
        # TOKEN COLUMNS
        # -------------------------------------------------

        t1, t2 = st.columns(2)


        with t1:

            st.markdown(
                "### Our Response Tokens"
            )

            our_token_df = pd.DataFrame(
                {
                    "Position": range(
                        1,
                        len(our_tokens) + 1
                    ),

                    "Token": our_tokens
                }
            )

            st.dataframe(
                our_token_df,
                use_container_width=True,
                hide_index=True
            )


        with t2:

            st.markdown(
                "### Reference Tokens"
            )

            reference_token_df = pd.DataFrame(
                {
                    "Position": range(
                        1,
                        len(reference_tokens) + 1
                    ),

                    "Token": reference_tokens
                }
            )

            st.dataframe(
                reference_token_df,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.warning(
            f"Tokenization display could not be loaded: {e}"
        )


    # =====================================================
    # RAW TENSOR OUTPUT
    # =====================================================

    st.subheader("🧮 Raw BERTScore Output")

    raw_df = pd.DataFrame(
        {
            "Metric": [
                "Precision",
                "Recall",
                "F1"
            ],

            "Tensor": [
                str(P_o),
                str(R_o),
                str(F1_o)
            ],

            "Value": [
                precision,
                recall,
                f1
            ]
        }
    )


    st.dataframe(
        raw_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.subheader("📝 Comparison Details")

    details_df = pd.DataFrame(
        {
            "Parameter": [
                "Language",
                "Model Selection",
                "Actual Model Argument",
                "IDF Weighting",
                "Baseline Rescaling"
            ],

            "Value": [
                lang,
                model_label,
                selected_model if selected_model else "BERTScore default",
                "No",
                "No"
            ]
        }
    )


    st.dataframe(
        details_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # TEXT DISPLAY
    # =====================================================

    st.subheader("📄 Input Text")

    d1, d2 = st.columns(2)


    with d1:

        st.markdown("**Our Response**")

        st.info(our_resp)


    with d2:

        st.markdown("**Reference**")

        st.info(reference)


    # =====================================================
    # FORMULA
    # =====================================================

    st.subheader("📐 BERTScore Components")

    st.markdown(
        """
        **Precision**

        Measures the semantic similarity of the candidate/
        response tokens against the reference.

        **Recall**

        Measures how well the reference tokens are captured
        by the candidate/response.

        **F1**

        The harmonic mean of Precision and Recall:

        `F1 = 2 × Precision × Recall / (Precision + Recall)`
        """
    )
