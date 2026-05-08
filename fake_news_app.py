import streamlit as st
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main {
    background: linear-gradient(135deg, #141E30, #243B55);
    padding: 2rem;
    border-radius: 20px;
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: white;
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    from {
        text-shadow: 0 0 10px #00C6FF;
    }
    to {
        text-shadow: 0 0 20px #0072FF;
    }
}

.subtitle {
    text-align: center;
    color: #d1d1d1;
    margin-bottom: 30px;
    font-size: 18px;
}

.stTextArea textarea {
    background-color: #1E1E1E;
    color: white;
    border-radius: 15px;
    border: 2px solid #00C6FF;
    font-size: 16px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #00C6FF, #0072FF);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(to right, #0072FF, #00C6FF);
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🧠 AI Fake News Detector</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Detect whether a news article is REAL or FAKE using Machine Learning & NLP</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT ----------------
news_text = st.text_area(
    "📰 Enter News Article",
    height=250,
    placeholder="Paste news article here..."
)

# ---------------- BUTTON ----------------
# ---------------- BUTTON ----------------
if st.button("🚀 Analyze News"):

    if news_text.strip() == "":
        st.warning("⚠ Please enter some news text.")

    else:

        with st.spinner("Analyzing with AI Model..."):

            news_vector = vectorizer.transform([news_text])

            prediction = model.predict(news_vector)

        if prediction[0] == 0:
            st.error("🚨 This News is FAKE")

        else:
            st.success("✅ This News is REAL")

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">Built with ❤️ using Machine Learning, NLP & Streamlit</div>',
    unsafe_allow_html=True
)
