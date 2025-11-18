##Customer Review Analytics: Sentiment Classification, Topic Extraction & Insights

An end-to-end **NLP and Machine Learning project** analyzing Amazon product reviews to extract:
- **Sentiment Classification (Positive/Negative)**
- **Topic Insights using LDA**
- **Summarized opinions**
- **Customer Satisfaction Metrics**
- **Model Deployment Proposal using AWS Lambda**

This project is designed as a **portfolio-ready ML case study** tailored to **Amazon Data Scientist Internship (2026)** requirements — demonstrating customer obsession, problem-solving, and deployment mindset.

---

## 1. Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| NLP | NLTK, Sumy |
| ML Models | Logistic Regression, SVM |
| Vectorization | TF-IDF |
| Topic Modeling | LDA (Latent Dirichlet Allocation) |
| Visualization | Matplotlib, Seaborn |
| Deployment Proposal | AWS Lambda + API Gateway |

---

## 2. Project Structure

```text 
Customer-Review-Analytics/
│
├── README.md
├── requirements.txt
├── train_small.csv
├── test_small.csv
├── svm_tfidf_model.joblib
├── tfidf_vectorizer.joblib
│
├── notebooks/
│   ├── 01_data_loading_and_preprocessing.ipynb
│   ├── 02_model_training_and_evaluation.ipynb
│   ├── 03_topic_extraction.ipynb
│   ├── 04_summarization.ipynb
│   └── 05_insights_and_demo.ipynb

```

---

## 3. Dataset

- Source: Amazon customer reviews dataset (downsampled for performance).
- `train_small.csv` → 100,000 processed reviews (balanced).
- `test_small.csv` → 10,000 reviews.
- Labels:  
  - `1` → Negative  
  - `2` → Positive

---

## 4. Feature Engineering

- Lowercase conversion  
- Removal of punctuation, digits, special characters  
- Tokenization & Stopword removal  
- TF-IDF vectorization  
- Topic modelling using n-grams & LDA

```python
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
```
---

## 5. Model Training Summary

- Models Tested:
  - Logistic Regression → ~89% accuracy
  - Linear SVM (Support Vector Machine) → **~90% accuracy (Best)**
- Vectorization:
  - TF-IDF (max_features=5000, ngram_range=(1,2))
- Final Model Selected:
  - LinearSVC (fast, scalable, good generalization)
- Model & Vectorizer Saved:
  ```python
  joblib.dump(svm_clf, "svm_tfidf_model.joblib")
  joblib.dump(tfidf, "tfidf_vectorizer.joblib")
   ```
---

## 6. Evaluation Metrics

```markdown
| Metric      | Value |
|------------|--------|
| Accuracy   | 0.90   |
| Precision  | 0.89   |
| Recall     | 0.90   |
| F1-score   | 0.89   |
| Test Size  | 10,000 reviews |

- Confusion Matrix showed balanced performance on both positive and negative reviews.
- SVM outperformed Logistic Regression and was chosen for deployment.

```
---

## 7. Topic Extraction

- Used Frequency Analysis (n-grams) + Latent Dirichlet Allocation (LDA)
- Common Positive Topics:
  - "highly recommend", "good value", "fast delivery"
- Common Negative Topics:
  - "poor quality", "late delivery", "refund process"

Example LDA Output:
Topic #0: ['value', 'worth', 'recommend', 'great', 'price']
Topic #1: ['late', 'shipping', 'delivery', 'delay', 'time']
Topic #2: ['quality', 'broke', 'cheap', 'issue', 'return']

---

## 8. Summarization

- Used TextRank Summarizer (via Sumy)
- Summary of Positive Reviews:
  "Customers appreciate fast delivery, quality packaging, and product reliability."

- Summary of Negative Reviews:
  "Common complaints include delayed shipping, weak build materials, and refund challenges."

- Helps management teams understand overall feedback without manually reading all reviews.

---

## 9. Deployment Plan (AWS Lambda)

📌 Proposed Architecture:
Client → API Gateway → AWS Lambda → Pre-trained Model → JSON Response

- Input: {"review": "The product is amazing"}
- Lambda loads:
  - "svm_tfidf_model.joblib"
  - "tfidf_vectorizer.joblib"
- Output: {"sentiment": "Positive"}

🔧 Suggested Steps:
1. Package model + vectorizer using joblib.
2. Create AWS Lambda (Python 3.9).
3. Integrate with API Gateway for POST requests.
4. Optional:
   - Store results in DynamoDB.
   - Trigger alerts for negative sentiment spikes.
