# 📝 Sentiment Analysis with Logistic Regression & BERT

This project predicts sentiment (positive, negative, neutral) from text using two approaches:
- Logistic Regression with TF-IDF (classical ML)
- BERT Transformer (state-of-the-art NLP)

## 🚀 Features
- Preprocessing tweets using TF-IDF
- Logistic Regression baseline model
- HuggingFace BERT sentiment classifier
- Deployed demo:
  - Logistic Regression → GitHub Pages
  - BERT → HuggingFace Spaces

## 📂 Project Structure
```

/notebooks        # Jupyter notebooks for training
/models           # Saved models (logreg + tfidf)
/app             # Frontend app for GitHub Pages
README.md         # Project documentation

````

## 🗂 Dataset
- [Twitter Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

## 🔧 Installation
```bash
pip install -r requirements.txt
````

## ▶️ Usage

Train Logistic Regression:

```bash
jupyter notebook notebooks/sentiment_baseline.ipynb
```

Run BERT sentiment classifier:

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
classifier("This flight was awesome!")
```

## 🌍 Deployment

* Logistic Regression demo hosted on GitHub Pages.
* BERT model deployed on HuggingFace Spaces.

## 📊 Results

* Logistic Regression: \~80% accuracy
* BERT: \~90%+ accuracy

```

---

✨ Deliverables after finishing:  
- GitHub repo with notebook + models.  
- GitHub Pages site with interactive demo.  
- HuggingFace Space (optional, but strong bonus).  