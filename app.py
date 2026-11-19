
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

# =====================================================
# APPLICATION CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="News Understanding System",
    page_icon="📰",
    layout="wide"
)

DISTILBERT_REPO = "Sameer63343/news-classifier-distilbert"
GPT2_REPO = "Sameer63343/news-summarizer-gpt2"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# LOAD MODELS FROM HUGGING FACE
# =====================================================

@st.cache_resource(show_spinner="Loading AI models from Hugging Face...")
def load_models():

    # DistilBERT
    classifier_tokenizer = AutoTokenizer.from_pretrained(
        DISTILBERT_REPO
    )

    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_REPO
    ).to(DEVICE)

    classifier_model.eval()

    print("\n========== MODEL CONFIGURATION ==========")

    print("Model:", classifier_model.config._name_or_path)

    print("ID to Label:")
    print(classifier_model.config.id2label)

    print("\nLabel to ID:")
    print(classifier_model.config.label2id)

    print("\nTokenizer:", classifier_tokenizer.name_or_path)

    print("=========================================\n")

    # GPT-2
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
# NEWS CLASSIFICATION
# =====================================================


def classify_news(news):

    # Check the original article's token count
    full_inputs = classifier_tokenizer(
        news,
        return_tensors="pt",
        truncation=False
    )

    original_tokens = full_inputs["input_ids"].shape[1]

    # Prepare model input
    inputs = classifier_tokenizer(
        news,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)

    processed_tokens = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = classifier_model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    predicted_id = probabilities.argmax(dim=-1).item()

    label = classifier_model.config.id2label[predicted_id]

    confidence = probabilities[0][predicted_id].item()

    # Diagnostic information in VS Code terminal
    print("\n========== CLASSIFIER DIAGNOSTICS ==========")

    print("Original token count:", original_tokens)
    print("Tokens processed:", processed_tokens)

    print("\nClass probabilities:")

    for i, probability in enumerate(probabilities[0]):
        category = classifier_model.config.id2label[i]

        print(
            f"{category}: {probability.item() * 100:.2f}%"
        )

    print("\nPredicted category:", label)
    print("Confidence:", f"{confidence * 100:.2f}%")

    print("============================================\n")

    return label, confidence





# =====================================================
# NEWS SUMMARY GENERATION
# =====================================================

def generate_summary(news):

    # Match the original CNN/DailyMail training format
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
# MAIN INTERFACE
# =====================================================

st.title("📰 News Understanding System")

st.markdown(
    """
    Analyze news articles using two fine-tuned
    Transformer models hosted on Hugging Face.

    **DistilBERT:** News classification

    **GPT-2:** News overview generation
    """
)

st.divider()

# Load models
try:

    (
        classifier_tokenizer,
        classifier_model,
        gpt2_tokenizer,
        gpt2_model
    ) = load_models()

    st.success("Both AI models are ready!")

except Exception as e:

    st.error(f"Unable to load models: {e}")
    st.stop()


# =====================================================
# NEWS INPUT
# =====================================================

st.subheader("Enter Your News Article")

news = st.text_area(
    "Paste your news article below:",
    height=250,
    placeholder="Enter or paste a complete news article..."
)

analyze = st.button(
    "Analyze News",
    type="primary",
    use_container_width=True
)


# =====================================================
# DISPLAY RESULTS
# =====================================================

if analyze:

    if not news.strip():

        st.warning("Please enter a news article first.")

    else:

        try:

            with st.spinner("Analyzing your news article..."):

                category, confidence = classify_news(news)

                summary = generate_summary(news)

            st.divider()

            st.header("Analysis Results")

            # Classification results
            col1, col2 = st.columns(2)

            with col1:

                st.subheader("News Category")

                st.markdown(
                    f"### {category}"
                )

            with col2:

                st.subheader("Classification Confidence")

                st.metric(
                    "Confidence Score",
                    f"{confidence * 100:.2f}%"
                )

                st.progress(confidence)

            st.divider()

            # Generated overview
            st.subheader("AI-Generated News Overview")

            if summary:

                st.write(summary)

            else:

                st.warning(
                    "The model did not generate a summary."
                )

            st.caption(
                "Powered by fine-tuned DistilBERT and GPT-2 models."
            )

        except Exception as e:

            st.error(f"Analysis failed: {e}")


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("About the System")

    st.write(
        "This application combines two fine-tuned "
        "Transformer models for automated news understanding."
    )

    st.divider()

    st.write("**Classification Model**")

    st.markdown(
        "[DistilBERT on Hugging Face]"
        "(https://huggingface.co/Sameer63343/news-classifier-distilbert)"
    )

    st.write("**Generation Model**")

    st.markdown(
        "[GPT-2 on Hugging Face]"
        "(https://huggingface.co/Sameer63343/news-summarizer-gpt2)"
    )

    st.divider()

    st.caption(f"Running on: {DEVICE}")
