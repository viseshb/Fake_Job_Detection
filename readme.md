
# Fake Job Detection 🚫💼

A lightweight, fully‑offline tool for spotting potentially fraudulent job postings by analyzing the **company profile** text. It combines classic NLP techniques (TF‑IDF) with supervised machine learning (Logistic Regression & Random Forest) and an interactive fuzzy‑matching layer so you can iteratively grow and refine the dataset.

---

## ✨ Key Features

| Feature                    | Description                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------- |
| **End‑to‑end pipeline**    | Cleans data → vectorizes text → trains models → makes predictions with probabilities              |
| **Bidirectional learning** | During prediction you can *add new company names* on‑the‑fly, immediately retraining the model    |
| **Two classifiers**        | Logistic Regression (balanced) & Random Forest (100 trees) saved as pickles for instant reuse     |
| **Fuzzy company matching** | Uses `fuzzywuzzy` to show the closest known company and a confidence score                        |
| **No cloud needed**        | Runs 100% locally; ideal for demos or coursework where internet may be restricted                 |
| **Supports short names**   | Now accepts 1-word inputs like "Google" or "Grok" with smart matching and prediction               |

---

## 🗂 Project Structure

```text
FAKE_JOB_DETECTION_PROJECT/
├── dataset/
│   └── Fake Postings.csv           # Main dataset (augmented with user entries)
├── model/
│   ├── known_companies.pkl         # Set of cleaned company profiles
│   ├── logistic_model.pkl          # Pickled Logistic Regression model
│   ├── random_forest_model.pkl     # Pickled Random Forest model
│   └── vectorizer.pkl              # Pickled TF‑IDF vectorizer
├── main.py                         # Entry point script (training + CLI)
├── requirements.txt                # Python dependencies
└── LICENSE                         # Project license (MIT)
```

---

## 🚀 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/Fake_Job_Detection_Project.git
   cd Fake_Job_Detection_Project
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the tool**
   ```bash
   python main.py
   ```

---

## 🖥️ Using the CLI

| Step                     | Action                                                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **Start‑up**             | Loads and cleans data. If no model exists, it trains from scratch.                                               |
| **Data Augmentation**    | Optionally add new company profiles (fraudulent/real) before predictions start. Retraining is automatic.         |
| **Prediction loop**      | Enter company names (e.g., "Google"). Tool will match against known profiles and predict fraud probability.       |
| **On‑the‑fly learning**  | If a company is missing, you can label it manually and it gets added and retrained instantly.                    |

### 🔍 Example

```text
🧠 Enter company profiles to detect fraud (blank to exit):
>> Google
🔹 Closest Match: google
🔹 Match Score: 100% → ✅ Highly confident match
🔹 Dataset Label: Real
🔹 Model Prediction: Real
🔹 Fraud Probability: 0.330
```

---

## 🔧 Configuration & Customization

| Variable       | Location           | Purpose                                                                 |
| -------------- | ------------------ | ----------------------------------------------------------------------- |
| `DATA_PATH`    | `main.py`          | Path to the dataset CSV file                                            |
| `max_features` | TF-IDF Vectorizer  | Adjust vocabulary richness                                              |
| `n_estimators` | Random Forest      | Number of decision trees                                                |

You can customize further with advanced models like `SVM`, `XGBoost`, or plug in `transformers`.

---

## 📊 Model Performance

- **Pre-processing**: Lowercasing + removing special characters
- **Vectorization**: TF-IDF (`max_features=3000`)
- **Class imbalance**: Automatically balanced using `class_weight='balanced'`
- **Training**: 80/20 stratified split unless a class has too few samples

Typical training output:

```
=== Logistic Regression ===
Accuracy: 99.5%
Confusion Matrix:
[[2002   19]
 [   0 2000]]
```

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- Dataset from the Kaggle community: *Fake Job Postings*
- Tools: `scikit-learn`, `pandas`, `numpy`, `fuzzywuzzy`

> *Built to help job seekers detect fraud with intelligence and simplicity.*

---
## 📉 Sample Output: Your Confusion Matrices
```
=== Logistic Regression ===
              precision    recall  f1-score   support

           0       1.00      0.99      1.00      2021
           1       0.99      1.00      1.00      2000

    accuracy                           1.00      4021
   macro avg       1.00      1.00      1.00      4021
weighted avg       1.00      1.00      1.00      4021

Confusion Matrix:
 [[2002   19]
 [   0 2000]]
Accuracy: 0.9952748072618751

=== Random Forest ===
              precision    recall  f1-score   support

           0       1.00      0.99      1.00      2021
           1       0.99      1.00      1.00      2000

    accuracy                           1.00      4021
   macro avg       1.00      1.00      1.00      4021
weighted avg       1.00      1.00      1.00      4021

Confusion Matrix:
 [[2002   19]
 [   0 2000]]
Accuracy: 0.9952748072618751
```
