import streamlit as st
import joblib

# Load saved model
model = joblib.load("models/fake_news_model.pkl")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Title
st.title("Fake News Detection App")

# Text input
news_text = st.text_area("Enter News Article")

# Predict button
if st.button("Check News"):

    # Convert text into vector
    news_vector = vectorizer.transform([news_text])

    # Predict
    prediction = model.predict(news_vector)

    # Output
    if prediction[0] == 0:
        st.error("🚨 This is Fake News")
    else:
        st.success("✅ This is Real News")
