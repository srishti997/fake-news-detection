**Fake News Detection Using Machine Learning & NLP**

This project builds a machine learning model that classifies news articles as Fake or Real using Natural Language Processing (NLP) techniques and a Logistic Regression classifier.
It achieves 99% accuracy on the Fake vs Real News dataset.

**Dataset**

Dataset Used: Fake and Real News Dataset (Kaggle)
The dataset contains:

Fake.csv → Fake news articles
True.csv → Real news articles
Both files were merged and labeled:

0 → Fake
1 → Real

Due to GitHub’s file-size limit, the dataset files (Fake.csv, True.csv) are not uploaded here.

You can download them from Kaggle:
👉 Fake and Real News Dataset
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset


**Project Workflow**
1. Data Loading
Loaded Fake.csv and True.csv using pandas.

2. Data Cleaning
Performed essential text preprocessing:
Lowercasing
Removing URLs
Removing numbers
Removing punctuation
Removing extra spaces

3. Feature Extraction (TF-IDF)
Converted cleaned text into numerical vectors using:
TfidfVectorizer
max_features = 5000
ngram_range = (1, 2)
stop_words = "english"

4. Model Training
Trained using PassiveAggressiveClassifier(max_iter=1000)
PAC works extremely well for:
Fake news detection
Real-time classification
Short news headlines
Online learning tasks

5. Model Evaluation
Evaluated the model using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Accuracy achieved: ≈ 99%

6. Custom Prediction Function
Created a function predict_news(text) that:
Cleans user input
Applies TF-IDF transformation
Predicts Fake/Real

**Results**
Metric	Fake	Real
Precision	0.99	0.99
Recall	0.99	0.99
F1-Score	0.99	0.99
Overall Accuracy: 99%
Confusion matrix shows very few misclassifications.

**Tech Stack Used**
Python
Pandas
NumPy
Scikit-learn
Regex
Google Colab

**How to Run This Project**
Option 1: Google Colab (Recommended)
Upload the notebook .ipynb
Upload Fake.csv and True.csv
Run all cells

Option 2: Local Machine 
pip install -r requirements.txt
jupyter notebook

**Author**
Srishti Gupta
B.Tech – Artificial Intelligence & Machine Learning

Email: srishtigupta997@gmail.com
