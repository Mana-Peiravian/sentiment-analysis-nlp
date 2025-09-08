# 📝 Sentiment Analysis with Logistic Regression & BERT

This project predicts sentiment (positive, negative, neutral) from text using two approaches:
- Logistic Regression with TF-IDF (classical ML)
- BERT Transformer (state-of-the-art NLP)

## 🚀 Features
- Preprocessing tweets using TF-IDF
- Logistic Regression baseline model
- HuggingFace BERT sentiment classifier
- Deployed demo:
  - Logistic Regression 
  - BERT 

## 📂 Project Structure
```

/notebooks        # Jupyter notebooks for training
/models           # Saved models (logreg + tfidf)
/docs             # Frontend app for GitHub Pages
/data             # Dataset from Kaggle
README.md         # Project documentation

````

## 🗂 Dataset
- [Twitter Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

## 🔧 Installation
```bash
pip install -r requirements.txt
```

## ▶️ Usage

Train Logistic Regression:

```bash
jupyter notebook notebooks/sentiment_baseline.ipynb
```

Run BERT sentiment classifier:

```bash
jupyter notebook notebooks/sentiment_bert.ipynb
```

## 🌍 Deployment

* Logistic Regression and BERT model demo hosted on GitHub Pages.

## 📊 Results

* Logistic Regression: \~80% accuracy
* BERT: \~90%+ accuracy
