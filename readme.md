# Fake Job Detection 🚫💼

A lightweight, fully‑offline tool for spotting potentially fraudulent job postings by analyzing the **company profile** text. It combines classic NLP techniques (TF‑IDF) with supervised machine‑learning (Logistic Regression & Random Forest) and an interactive fuzzy‑matching layer so you can iteratively grow and refine the dataset.

---

## ✨ Key Features

| Feature                    | Description                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------- |
| **End‑to‑end pipeline**    | Cleans data → vectorizes text → trains models → makes predictions with probabilities              |
| **Bidirectional learning** | During prediction you can *add new company profiles* on‑the‑fly, immediately retraining the model |
| **Two classifiers**        | Logistic Regression (balanced) & Random Forest (100 trees) saved as pickles for instant reuse     |
| **Fuzzy lookup**           | Uses `fuzzywuzzy` to show the closest match in the dataset and a confidence score                 |
| **No cloud needed**        | Runs 100 % locally; ideal for demos or coursework where internet may be restricted                |

---

## 🗂 Project Structure

```text
FAKE_JOB_DETECTION_PROJECT/
├── dataset/
│   └── Fake Postings.csv           # Kaggle dataset (raw source)
├── model/                          # Auto‑generated after the first run
│   ├── known_companies.pkl         # Set of cleaned company profiles
│   ├── logistic_model.pkl          # Pickled Logistic Regression model
│   ├── random_forest_model.pkl     # Pickled Random Forest model
│   └── vectorizer.pkl              # Pickled TF‑IDF vectorizer
├── main.py                         # Entry‑point script (train + predict loop)
├── requirements.txt                # Python dependencies
└── LICENSE                         # Project license (MIT )
```

---

## 🚀 Quick Start

1. **Clone & enter the repo**
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/Fake_Job_Detection_Project.git
   cd Fake_Job_Detection_Project
   ```
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app**
   ```bash
   python main.py
   ```
   *First launch trains the models and writes them to **`model/`**. Subsequent runs load the pickles instantly.*

---

## 🖥️ Using the CLI

| Step                     | What happens                                                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Start‑up**             | Dataset is cleaned (`regex` + lowercase). If no pickles are found, models are trained from scratch                                                |
| **Optional data entry**  | You can add new company profiles *before* predictions begin; each addition triggers instant retraining                                            |
| **Prediction loop**      | Paste any company‑profile text → the tool prints: cleaned text, closest dataset match, fuzzy score, dataset label, model prediction & probability |
| **Interactive learning** | Unknown companies can be labelled on the spot (fraudulent / real) and appended to the CSV, then the model is retrained                            |

### Example Session

```text
🔄 Loading and preparing dataset…
✅ Loaded existing model.

Would you like to add any new company profiles? (yes/no): no

🧠 Enter company profiles to detect fraud (blank to exit):
>> Leading FinTech firm providing zero‑fee credit cards worldwide
🔹 Closest Match: innovative fintech startup disrupting global payments
🔹 Match Score: 86 % → ⚠️ Acceptable match – use with caution
🔹 Dataset Label: Real
🔹 Model Prediction: Fraudulent
🔹 Fraud Probability: 0.78
--------------------------------------------------
```

---

## 🔧 Configuration & Customization

| Variable       | Where             | Purpose                                                              |
| -------------- | ----------------- | -------------------------------------------------------------------- |
| `DATA_PATH`    | `main.py`         | Absolute path to `Fake Postings.csv`. Adjust if you move the dataset |
| `max_features` | TF‑IDF vectorizer | Increase for richer vocabulary (may slow training)                   |
| `n_estimators` | Random Forest     | Tune tree count for accuracy vs. speed                               |

Feel free to fork the repo and experiment with alternative models (e.g. SVM, XGBoost) or more sophisticated text cleaning (stemming, stop‑word removal).

---

## 📊 Model Details

- **Pre‑processing**: Non‑word characters removed, lower‑cased, extra spaces trimmed.
- **Vectorization**: TF‑IDF with a maximum of 3 000 features.
- **Class imbalance**: `class_weight='balanced'` for Logistic Regression; balanced subsampling is built‑in for Random Forest.
- **Evaluation**: 80 / 20 stratified split (unless a class has fewer than two samples, in which case the model trains on the full dataset).

Sample metrics printed after training:

```
=== Random Forest ===
precision    recall    f1‑score    support
0     0.97     0.99       0.98       6668
1     0.81     0.45       0.58        378
accuracy                0.97       7046
```

---

## 🤝 Contributing

Pull requests are welcome! If you spot a bug or have an idea for improvement:

1. Open an issue describing the change.
2. Create a feature branch (`git checkout -b feature/my‑feature`).
3. Commit your changes with clear messages.
4. Push and open a PR.

Please run linting/tests before submitting (coming soon).

---

## 📄 License

This project is released under the **MIT License** – see the `LICENSE` file for details.

---

## 🙏 Acknowledgements

- Original *Fake Job Postings* dataset – © Kaggle community contributors.
- `scikit‑learn`, `pandas`, `numpy`, and `fuzzywuzzy` – the open‑source workhorses powering this project.

> *Built with ❤ to help job‑seekers stay safe.*

