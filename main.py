from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# === Paths ===
DATA_PATH = 'C:/Users/vises/OneDrive/Desktop/Masters_TAMUSA/Sem2/Artificial Intelligence/Fake_Job_Detection_Project/dataset/Fake Postings.csv'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# === Text Cleaning ===
def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r'\W+', ' ', str(text)).lower().strip()

# === Load Dataset ===
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df['company_profile_clean'] = df['company_profile'].apply(clean_text)
    df = df[df['company_profile_clean'] != ""]
    return df


def train_model(df):
    df['label'] = df['fraudulent']
    df = df[['company_profile_clean', 'label']]

    print("✅ Final class distribution:\n", df['label'].value_counts())

    value_counts = df['label'].value_counts()
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['company_profile_clean'])
    y = df['label']

    if len(value_counts) < 2 or value_counts.min() < 2:
        print("⚠️ Not enough samples in each class. Training on entire data without test split.")

        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        lr.fit(X, y)
        print("\n=== Logistic Regression ===")
        print("Trained on full data (no test split).")

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        print("\n=== Random Forest ===")
        print("Trained on full data (no test split).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # === Logistic Regression ===
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        print("\n=== Logistic Regression ===")
        print(classification_report(y_test, y_pred_lr, zero_division=0))
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        print("Confusion Matrix:\n", cm_lr)
        print("Accuracy:", accuracy_score(y_test, y_pred_lr))

        # Save confusion matrix image for Logistic Regression
        disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Real', 'Fake'])
        disp_lr.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title("Logistic Regression - Confusion Matrix")
        plt.savefig("images/confusion_matrix_lr.png")
        plt.close()

        # === Random Forest ===
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        print("\n=== Random Forest ===")
        print(classification_report(y_test, y_pred_rf, zero_division=0))
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        print("Confusion Matrix:\n", cm_rf)
        print("Accuracy:", accuracy_score(y_test, y_pred_rf))

        # Save confusion matrix image for Random Forest
        disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Real', 'Fake'])
        disp_rf.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title("Random Forest - Confusion Matrix")
        plt.savefig("images/confusion_matrix_rf.png")
        plt.close()
    pickle.dump(lr, open(f"{MODEL_DIR}/logistic_model.pkl", "wb"))
    pickle.dump(rf, open(f"{MODEL_DIR}/random_forest_model.pkl", "wb"))
    pickle.dump(vectorizer, open(f"{MODEL_DIR}/vectorizer.pkl", "wb"))
    pickle.dump(set(df['company_profile_clean']), open(f"{MODEL_DIR}/known_companies.pkl", "wb"))
    return rf, vectorizer

# === Fuzzy Match + Prediction ===
def interpret_match_score(score):
    if score >= 90:
        return "✅ Highly confident match"
    elif 80 <= score < 90:
        return "⚠️ Acceptable match – use with caution"
    else:
        return "❌ Too weak – typically unreliable"

def predict_company_profile_advanced(text, model, vectorizer, full_dataset):
    cleaned = clean_text(text)

    if len(cleaned) < 3:
        print(f"\n🔍 Input: {text}")
        print("⚠️ Input too short. Please provide at least 3 characters.")
        print("-" * 50)
        return False

    known_profiles = full_dataset['company_profile_clean'].tolist()
    best_match, match_score = process.extractOne(cleaned, known_profiles, scorer=fuzz.token_sort_ratio)

    if match_score is None or match_score < 70:
        print(f"\n🔍 Input: {text}")
        print(f"🔹 Cleaned: {cleaned}")
        print("❌ Company not found in dataset.")
        add = input("➕ Do you want to add this company to the dataset? (yes/no): ").strip().lower()
        if add == 'yes':
            while True:
                label_input = input("Is this company fraudulent? (yes/no): ").strip().lower()
                if label_input in ['yes', 'no']:
                    break
                print("Please enter 'yes' or 'no'.")
            fraud_val = 1 if label_input == 'yes' else 0

            new_row = {
                'title': '',
                'description': '',
                'requirements': '',
                'company_profile': text,
                'location': '',
                'salary_range': '',
                'employment_type': '',
                'industry': '',
                'benefits': '',
                'fraudulent': fraud_val
            }

            df_existing = full_dataset.drop(columns=['company_profile_clean'])
            df_updated = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
            df_updated.to_csv(DATA_PATH, index=False)

            print("✅ Company added. Retraining model...\n")
            new_df = load_dataset()
            new_model, new_vectorizer = train_model(new_df)

            # Re-run prediction
            df_updated = load_dataset()
            predict_company_profile_advanced(text, new_model, new_vectorizer, df_updated)
            return True
        else:
            print("ℹ️ Skipped. Not added.")
            return False

    # Found match → Predict
    label_row = full_dataset[full_dataset['company_profile_clean'] == best_match]
    dataset_label = int(label_row['fraudulent'].values[0]) if not label_row.empty else None

    X_new = vectorizer.transform([cleaned])
    prob = model.predict_proba(X_new)[0][1]
    pred = model.predict(X_new)[0]

    print(f"\n🔍 Input: {text}")
    print(f"🔹 Cleaned: {cleaned}")
    print(f"🔹 Closest Match: {best_match}")
    print(f"🔹 Match Score: {match_score}% → {interpret_match_score(match_score)}")
    print(f"🔹 Dataset Label: {'Fraudulent' if dataset_label == 1 else 'Real'}")
    print(f"🔹 Model Prediction: {'Fraudulent' if pred == 1 else 'Real'}")
    print(f"🔹 Fraud Probability: {prob:.3f}")
    print("-" * 50)
    return True

# === Main ===
def main():
    print("🔄 Loading and preparing dataset...")
    df = load_dataset()

    # Try loading model
    try:
        rf_model = pickle.load(open(f"{MODEL_DIR}/random_forest_model.pkl", "rb"))
        vectorizer = pickle.load(open(f"{MODEL_DIR}/vectorizer.pkl", "rb"))
        print("✅ Loaded existing model.")
    except:
        print("⚙️ No model found. Training a new one...")
        rf_model, vectorizer = train_model(df)

    # Prompt for manual additions
    add_new = input("\nWould you like to add any new company profiles? (yes/no): ").strip().lower()
    if add_new == 'yes':
        new_rows = []
        print("Enter new company profiles (leave blank to finish):")
        while True:
            new_profile = input(">> ").strip()
            if new_profile == "":
                break
            cleaned = clean_text(new_profile)
            if cleaned in df['company_profile_clean'].values:
                print("⚠️ Already exists. Skipping.")
                continue

            while True:
                label_input = input("Is this company fraudulent? (yes/no): ").strip().lower()
                if label_input in ['yes', 'no']:
                    break
                print("Please enter 'yes' or 'no'.")
            fraud_val = 1 if label_input == 'yes' else 0
            new_rows.append({
                'title': '', 'description': '', 'requirements': '', 'company_profile': new_profile,
                'location': '', 'salary_range': '', 'employment_type': '',
                'industry': '', 'benefits': '', 'fraudulent': fraud_val
            })

        if new_rows:
            print(f"📝 Adding {len(new_rows)} new entries to dataset and retraining...")
            df_existing = df.drop(columns=['company_profile_clean'])
            df_updated = pd.concat([df_existing, pd.DataFrame(new_rows)], ignore_index=True)
            df_updated.to_csv(DATA_PATH, index=False)
            df = load_dataset()
            rf_model, vectorizer = train_model(df)

    # Prediction loop
    print("\n🧠 Enter company profiles to detect fraud (blank to exit):")
    while True:
        user_input = input(">> ").strip()
        if user_input == "":
            print("👋 Exiting.")
            break

        # ✅ Always reload latest dataset and model after any additions
        df = load_dataset()
        rf_model = pickle.load(open(f"{MODEL_DIR}/random_forest_model.pkl", "rb"))
        vectorizer = pickle.load(open(f"{MODEL_DIR}/vectorizer.pkl", "rb"))

        predict_company_profile_advanced(user_input, rf_model, vectorizer, df)


if __name__ == "__main__":
    main()