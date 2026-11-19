
import html

import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="News AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISTILBERT_REPO = "Sameer63343/news-classifier-distilbert"
GPT2_REPO = "Sameer63343/news-summarizer-gpt2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# DESIGN / CSS
# =====================================================

st.html("""
<style>

/* Hide Streamlit's white header and top toolbar */
[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Main background */
.stApp,
[data-testid="stAppViewContainer"] {
    background: #08111F !important;
    color: #EAF2FF !important;
}

/* Remove unnecessary top spacing */
.block-container {
    max-width: 1150px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #101C2D !important;
    border-right: 1px solid #263B54;
}

[data-testid="stSidebar"] * {
    color: #EAF2FF;
}

/* Headings and text */
h1, h2, h3 {
    color: #F4F8FF !important;
}

p, label, .stMarkdown {
    color: #C0CCDC;
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #122641, #123B50);
    border: 1px solid #285276;
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 28px;
}

.hero-tag {
    color: #70D7FF;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 15px;
}

.hero-title {
    color: #FFFFFF;
    font-size: 34px;
    font-weight: 800;
    line-height: 1.3;
}

.hero-title span {
    color: #65D5FF;
}

.hero-description {
    color: #C3D3E5;
    font-size: 15px;
    margin-top: 12px;
    line-height: 1.6;
}

/* Model cards */
.model-card {
    background: #17263A;
    border: 1px solid #2A405C;
    border-radius: 12px;
    padding: 15px;
    margin-top: 12px;
}

.model-name {
    color: #F0F6FF;
    font-size: 16px;
    font-weight: 700;
}

.model-purpose {
    color: #70D7FF;
    font-size: 11px;
    margin-top: 5px;
}

/* Result cards */
.result-card {
    background: #14243A;
    border: 1px solid #2B4969;
    border-radius: 14px;
    padding: 22px;
    min-height: 145px;
}

.card-caption {
    color: #A9BDD4;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.category-value {
    color: #70D7FF;
    font-size: 28px;
    font-weight: 800;
    margin-top: 15px;
}

.confidence-value {
    color: #70E0B8;
    font-size: 28px;
    font-weight: 800;
    margin-top: 15px;
}

.summary-card {
    background: #14243A;
    border: 1px solid #2B4969;
    border-radius: 14px;
    padding: 22px;
    color: #EAF2FF;
    font-size: 15px;
    line-height: 1.8;
    white-space: pre-wrap;
}

/* Input */
textarea {
    background: #14243A !important;
    color: #FFFFFF !important;
    border: 1px solid #38516E !important;
    border-radius: 10px !important;
}

textarea::placeholder {
    color: #91A5BC !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #168BC7, #286CE0);
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #20A6E5, #4386F4);
    color: #FFFFFF !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: #21A9DF;
}

/* Divider */
hr {
    border-color: #263B54;
}

/* Expander */
[data-testid="stExpander"] {
    background: #101C2D;
    border: 1px solid #263B54;
    border-radius: 10px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""")


# =====================================================
# LOAD MODELS FROM HUGGING FACE
# =====================================================

@st.cache_resource(
    show_spinner="Loading AI models from Hugging Face..."
)
def load_models():

    classifier_tokenizer = AutoTokenizer.from_pretrained(
        DISTILBERT_REPO
    )

    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_REPO
    ).to(DEVICE)

    classifier_model.eval()

    gpt2_tokenizer = AutoTokenizer.from_pretrained(
        GPT2_REPO
    )

    gpt2_model = AutoModelForCausalLM.from_pretrained(
        GPT2_REPO
    ).to(DEVICE)

    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    gpt2_model.config.pad_token_id = gpt2_tokenizer.pad_token_id

    gpt2_model.eval()

    return (
        classifier_tokenizer,
        classifier_model,
        gpt2_tokenizer,
        gpt2_model
    )


# =====================================================
# CLASSIFICATION
# =====================================================

def classify_news(news):

    inputs = classifier_tokenizer(
        news,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)

    with torch.inference_mode():

        outputs = classifier_model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    predicted_id = probabilities.argmax(dim=-1).item()

    label = classifier_model.config.id2label[predicted_id]

    confidence = probabilities[0][predicted_id].item()

    all_probabilities = {}

    for i, probability in enumerate(probabilities[0]):

        category = classifier_model.config.id2label[i]

        all_probabilities[category] = probability.item()

    return label, confidence, all_probabilities


# =====================================================
# OVERVIEW GENERATION
# =====================================================

def generate_summary(news):

    prompt = (
        f"Article:\n{news}\n\n"
        "Detailed summary:\n"
    )

    inputs = gpt2_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=610
    ).to(DEVICE)

    with torch.inference_mode():

        outputs = gpt2_model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=False,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            eos_token_id=gpt2_tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    summary = gpt2_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return summary.strip()


# =====================================================
# LOAD MODELS
# =====================================================

try:

    (
        classifier_tokenizer,
        classifier_model,
        gpt2_tokenizer,
        gpt2_model
    ) = load_models()

except Exception as e:

    st.error(f"Unable to load AI models: {e}")

    st.stop()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 📰 NEWS AI")

    st.caption("NEWS INTELLIGENCE")

    st.divider()

    st.markdown("### AI MODELS")

    st.html("""
    <div class="model-card">
        <div class="model-name">🧠 DistilBERT</div>
        <div class="model-purpose">NEWS CLASSIFICATION</div>
    </div>

    <div class="model-card">
        <div class="model-name">✨ GPT-2</div>
        <div class="model-purpose">NEWS OVERVIEW GENERATION</div>
    </div>
    """)

   




# =====================================================
# HERO SECTION
# =====================================================

st.html("""
<div class="hero">

    <div class="hero-tag">
        AI-POWERED NEWS ANALYSIS
    </div>

    <div class="hero-title">
        Understand the News.<br>
        <span>Powered by AI.</span>
    </div>

    <div class="hero-description">
        Classify news articles and generate detailed
        overviews using fine-tuned AI models.
    </div>

</div>
""")


# =====================================================
# NEWS INPUT
# =====================================================

st.subheader("📰 Analyze a News Article")

st.caption(
    "Paste your news article below to begin."
)

news = st.text_area(
    "News article",
    height=250,
    placeholder="Paste your complete news article here...",
    label_visibility="collapsed",
    key="news_input"
)

word_count = len(news.split()) if news.strip() else 0

col1, col2 = st.columns([3, 1])

with col1:

    st.caption(f"{word_count} words")

with col2:

    analyze = st.button(
        "✦ ANALYZE NEWS",
        type="primary",
        use_container_width=True
    )


# =====================================================
# ANALYSIS RESULTS
# =====================================================

if analyze:

    if not news.strip():

        st.warning("Please enter a news article first.")

    else:

        try:

            with st.spinner("Analyzing your news article..."):

                category, confidence, probabilities = classify_news(news)

                summary = generate_summary(news)

            st.divider()

            st.subheader("Analysis Results")

            # Classification icons
            category_icons = {
                "World": "🌍",
                "Sports": "🏆",
                "Business": "📈",
                "Sci/Tech": "🔬"
            }

            icon = category_icons.get(category, "📰")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.html(f"""
                <div class="result-card">

                    <div class="card-caption">
                        NEWS CATEGORY
                    </div>

                    <div class="category-value">
                        {html.escape(icon)} {html.escape(str(category))}
                    </div>

                </div>
                """)

            with result_col2:

                st.html(f"""
                <div class="result-card">

                    <div class="card-caption">
                        CONFIDENCE
                    </div>

                    <div class="confidence-value">
                        {confidence * 100:.2f}%
                    </div>

                </div>
                """)

            # Probability breakdown
            with st.expander("Classification details"):

                for label, probability in probabilities.items():

                    st.markdown(
                        f"**{label}** — {probability * 100:.2f}%"
                    )

                    st.progress(probability)

            st.divider()

            # Generated overview
            st.subheader("✨ Generated News Overview")

            if summary:

                safe_summary = html.escape(summary)

                st.html(
                    f'<div class="summary-card">{safe_summary}</div>'
                )

            else:

                st.warning(
                    "The model did not generate an overview."
                )

            st.caption(
                "AI-generated information may contain inaccuracies. "
                "Verify important facts with the original article."
            )

            st.success("Analysis completed successfully.")

        except Exception as e:

            st.error(f"Analysis failed: {e}")
