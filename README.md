# Fake News Detection System 📰

A Machine Learning + NLP web application that detects whether a news article is Fake News or Real News.

Built using:
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Machine Learning Models
- Streamlit Deployment

---

# 🚀 Project Overview

This project analyzes news article text and predicts whether the news is fake or real.

The system uses:
- text preprocessing
- stopword removal
- lemmatization
- TF-IDF vectorization
- classification models

---

# 🧠 NLP Pipeline

Raw Text
↓
Text Cleaning
↓
Lowercasing
↓
Stopword Removal
↓
Lemmatization
↓
TF-IDF Vectorization
↓
Machine Learning Model
↓
Prediction

---

# 📊 Models Used

| Model | Accuracy |
|---|---|
| Logistic Regression | 98.8% |
| Naive Bayes | 93.0% |
| Passive Aggressive Classifier | 99.4% |

The Passive Aggressive Classifier achieved the best performance and was selected as the final model.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Joblib

---

# 📂 Project Structure

## 📂 Project Structure

```text
fake-news-detector/

├── app.py
├── requirements.txt
├── README.md

├── data/

├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl

├── notebooks/
│   └── fake_news_training.ipynb
```

---

# 🌐 Features

- Fake vs Real News Detection
- NLP preprocessing pipeline
- Real-time prediction
- Interactive Streamlit web app
- Multiple model comparison

---

# ▶️ Run Locally

bash id="n6pqdf" pip install -r requirements.txt streamlit run app.py 

---

# 📌 Future Improvements

- Better UI/UX
- Deep Learning NLP models
- BERT/Transformer integration
- Multi-language fake news detection

---

# 👨‍💻 Author

Shashank s
