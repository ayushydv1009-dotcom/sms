import streamlit as st
import numpy as np
import re, os, pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Constants (must match train.py) ──────────────────────────────
MAX_LEN = 100

def clean_text(text):
    """Lowercase, strip special characters and extra whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_model():
    """Load the pre-trained Keras model (cached so it loads only once)."""
    path = 'spam_model.h5'
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

@st.cache_resource
def load_tokenizer():
    """Load the fitted tokenizer saved during training."""
    path = 'tokenizer.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def predict_sms(model, tokenizer, message):
    """Return (label, confidence) for a single SMS message."""
    cleaned = clean_text(message)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob = model.predict(padded, verbose=0)[0][0]
    label = 'SPAM' if prob >= 0.5 else 'HAM'
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence

def main():
    st.set_page_config(page_title="SMS Spam Detector", layout="wide")
    st.title("SMS Spam Detector")
    st.markdown("Type or paste an SMS message below to check if it's **spam** or **ham (legitimate)**.")

    # ── Load artifacts ───────────────────────────────────────────
    model = load_model()
    tokenizer = load_tokenizer()

    if model is None or tokenizer is None:
        st.error(
            "Pre-trained model (`spam_model.h5`) or tokenizer (`tokenizer.pkl`) not found.\n\n"
            "Please run `python train.py` first to generate them."
        )
        return

    st.sidebar.success("Model & tokenizer loaded successfully!")

    # ── User input ───────────────────────────────────────────────
    user_msg = st.text_area("Enter SMS message", height=120,
                            placeholder="e.g. Congratulations! You won a free iPhone …")

    if st.button("Predict"):
        if not user_msg.strip():
            st.warning("Please enter a message first.")
        else:
            label, confidence = predict_sms(model, tokenizer, user_msg)

            col1, col2 = st.columns(2)
            with col1:
                colour = "#e74c3c" if label == "SPAM" else "#2ecc71"
                st.markdown(
                    f"<h1 style='text-align:center; color:{colour};'>{label}</h1>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"<h3 style='text-align:center;'>Confidence: {confidence:.2%}</h3>",
                    unsafe_allow_html=True,
                )

    # ── Sample predictions ───────────────────────────────────────
    st.divider()
    st.subheader("Try Sample Messages")

    samples = [
        "Congratulations! You have won a free iPhone. Click here to claim now!",
        "Hey, are you coming to the meeting at 3pm today?",
        "URGENT! Your bank account has been suspended. Call 08001234 immediately.",
    ]

    for msg in samples:
        with st.expander(f"📩  {msg[:60]}…" if len(msg) > 60 else f"📩  {msg}"):
            lbl, conf = predict_sms(model, tokenizer, msg)
            colour = "#e74c3c" if lbl == "SPAM" else "#2ecc71"
            st.markdown(f"**Prediction:** <span style='color:{colour}'>{lbl}</span> &nbsp; | &nbsp; **Confidence:** {conf:.2%}",
                        unsafe_allow_html=True)

    # ── Evaluation artifacts ─────────────────────────────────────
    st.divider()
    st.subheader("Model Evaluation Artifacts (from Training)")

    col3, col4 = st.columns(2)

    with col3:
        if os.path.exists('training_curves.png'):
            st.image('training_curves.png', caption="Training Curves", use_container_width=True)
        else:
            st.info("Training curves plot not generated yet.")

    with col4:
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix plot not generated yet.")

if __name__ == "__main__":
    main()
