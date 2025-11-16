import streamlit as st
import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ---------- Text cleaning function ----------
def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove URLs, numbers, punctuation, extra spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"\d+", "", text)                     # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()            # remove extra spaces
    return text


# ---------- Load data + train model (cached so it only runs once) ----------
@st.cache_resource
def load_model():
    # Load CSVs (make sure Fake.csv and True.csv are in the same folder as app.py)
    fake_df = pd.read_csv("Fake.csv")
    true_df = pd.read_csv("True.csv")

    # Add labels
    fake_df["label"] = 0   # Fake
    true_df["label"] = 1   # Real

    # Combine title + text
    fake_df["content"] = fake_df["title"].astype(str) + " " + fake_df["text"].astype(str)
    true_df["content"] = true_df["title"].astype(str) + " " + true_df["text"].astype(str)

    # Keep only what we need
    fake_df = fake_df[["content", "label"]]
    true_df = true_df[["content", "label"]]

    # Merge and shuffle
    data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Clean text
    data["clean_content"] = data["content"].apply(clean_text)

    X = data["clean_content"]
    y = data["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Optional: you can compute accuracy here if you want
    # from sklearn.metrics import accuracy_score
    # acc = accuracy_score(y_test, model.predict(X_test_tfidf))
    # print("Accuracy:", acc)

    return model, tfidf


# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

    st.title("📰 Fake News Detection App")
    st.write(
        "This app uses **Machine Learning + NLP** to classify news text as "
        "**Fake** or **Real**."
    )

    # Load model (cached, so this won't re-run every time)
    with st.spinner("Loading model and vectorizer..."):
        model, tfidf = load_model()
    st.success("Model loaded successfully ✅")

    st.markdown("---")

    st.subheader("🔍 Enter news text to classify")

    user_input = st.text_area(
        "Paste a news headline or article below:",
        height=200,
        placeholder="Example: Government announces new education policy for rural schools..."
    )

    if st.button("Predict"):
        if not user_input.strip():
            st.error("Please enter some text before clicking Predict.")
        else:
            # Clean and transform
            cleaned = clean_text(user_input)
            vec = tfidf.transform([cleaned])

            # Predict
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]

            label = "REAL ✅" if pred == 1 else "FAKE ❌"

            st.markdown("### 🧾 Prediction")
            st.write(f"**This news article looks: {label}**")

            st.markdown("### 📊 Confidence")
            st.write(f"- Real: `{proba[1]*100:.2f}%`")
            st.write(f"- Fake: `{proba[0]*100:.2f}%`")

            st.markdown("---")
            st.caption("Note: This is a student project model trained on a specific dataset; "
                       "it should not be used as a real-world fact-checking tool.")


    # Sidebar info
    st.sidebar.title("About")
    st.sidebar.info(
        "This is a mini project built by a B.Tech AIML student.\n\n"
        "Model: Logistic Regression\n"
        "Features: TF-IDF (unigrams + bigrams)\n"
        "Dataset: Fake and Real News (Kaggle)"
    )


if __name__ == "__main__":
    main()
