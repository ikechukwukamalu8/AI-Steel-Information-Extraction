"""
AI-STEEL: Information Extraction & Sustainable Property Prediction
Author: Kamalu Ikechukwu
Description: End-to-end pipeline for extracting metallurgical knowledge 
             from legacy PDFs, merging datasets, and ML-based prediction.
"""

import os
import re
import pdfplumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURATION ---
DATA_DIR = "data"
RESULTS_DIR = "results"
MAIN_DATA_PATH = os.path.join(DATA_DIR, "final_steel_data.csv")
PDF_FILES = ["Old Laboratory Notebook.pdf", "Metallurgical Study – AISI Steel Grades.pdf"]
OUTPUT_CSV = os.path.join(RESULTS_DIR, "ai_steel_combined_master.csv")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------
# 1. SCIENTIFIC KNOWLEDGE EXTRACTION
# -----------------------------------------------------------
def extract_samples_from_text(text, filename):
    samples = []
    # Split text by "Specimen" or "Sample" to isolate different records
    chunks = re.split(r"(Specimen:|Sample [A-Z])", text)
    
    for chunk in chunks:
        if len(chunk) < 20: continue 
        
        # Enhanced Regex: Handles LaTeX notation, tildes, and approximation symbols
        grade = re.search(r"(?:Grade|steel)\s*(\d{4})", chunk, re.I)
        carbon = re.search(r"(?:Carbon|C)[\s\approx\~:]*(\d?\.\d+)", chunk, re.I)
        manganese = re.search(r"(?:Manganese|Mn)[\s\approx\~:]*(\d?\.\d+)", chunk, re.I)
        quench = re.search(r"(?:quench|at)[\s\approx\~]*(\d{3,4})", chunk, re.I)
        temper = re.search(r"temper(?:ing)?.*?(\d+\.?\d*)\s?hr", chunk, re.I)
        strength = re.search(r"(\d{3,4})\s?MPa", chunk, re.I)

        if quench or strength:
            samples.append({
                "grade": grade.group(1) if grade else "Unknown",
                "carbon_pct": float(carbon.group(1)) if carbon else None,
                "manganese_pct": float(manganese.group(1)) if manganese else None,
                "quench_temp": float(quench.group(1)) if quench else None,
                "temper_time": float(temper.group(1)) if temper else None,
                "tensile_strength": float(strength.group(1)) if strength else None,
                "source": filename
            })
    return samples

# -----------------------------------------------------------
# 2. SUSTAINABILITY MODULE (Energy & CO2 Estimation)
# -----------------------------------------------------------
def calculate_carbon_footprint(temp_c):
    """Calculates CO2 emissions based on quenching energy requirements."""
    if pd.isna(temp_c): return 0
    # Constants for 1kg steel: specific heat (0.49), Efficiency (0.6), Grid Factor (0.4)
    energy_kwh = (1.0 * 0.49 * (temp_c - 25)) / (0.6 * 3600)
    return energy_kwh * 0.4

# -----------------------------------------------------------
# 3. DATA INTEGRATION PIPELINE
# -----------------------------------------------------------
def prepare_dataset():
    print("--- 📂 Extraction Phase ---")
    all_extracted = []
    
    for pdf_name in PDF_FILES:
        path = os.path.join(DATA_DIR, pdf_name)
        if os.path.exists(path):
            with pdfplumber.open(path) as pdf:
                full_text = "\n".join([p.extract_text() or "" for p in pdf.pages])
                found = extract_samples_from_text(full_text, pdf_name)
                all_extracted.extend(found)
                print(f"Extracted {len(found)} samples from {pdf_name}")

    pdf_df = pd.DataFrame(all_extracted)
    
    # Ensure all columns exist before merge
    required = ['grade', 'carbon_pct', 'manganese_pct', 'quench_temp', 'temper_time', 'tensile_strength']
    for col in required:
        if col not in pdf_df.columns: pdf_df[col] = np.nan

    pdf_df = pdf_df.dropna(subset=['tensile_strength'])

    print("\n--- 📂 Merging & Imputation Phase ---")
    if os.path.exists(MAIN_DATA_PATH):
        main_df = pd.read_csv(MAIN_DATA_PATH)
        combined_df = pd.concat([main_df, pdf_df], ignore_index=True)
        
        # Intelligent Imputation: Fill missing chemistry based on Grade averages
        combined_df['carbon_pct'] = combined_df.groupby('grade')['carbon_pct'].transform(lambda x: x.fillna(x.mean()))
        combined_df['manganese_pct'] = combined_df.groupby('grade')['manganese_pct'].transform(lambda x: x.fillna(x.mean()))
        combined_df = combined_df.fillna(combined_df.mean(numeric_only=True))
        
        # Apply Sustainability Metric
        combined_df['co2_kg'] = combined_df['quench_temp'].apply(calculate_carbon_footprint)
        
        combined_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Master dataset saved to: {OUTPUT_CSV}")
        return combined_df
    return pdf_df

# -----------------------------------------------------------
# 4. ML MODELING & EVALUATION
# -----------------------------------------------------------
def train_and_analyze(df):
    print("\n--- 🤖 Machine Learning Phase ---")
    features = ["carbon_pct", "manganese_pct", "quench_temp", "temper_time"]
    X = df[features]
    y = df["tensile_strength"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    print(f"Model Performance: R2 Score = {r2_score(y_test, preds):.4f}")

    # Visualizing Prediction Accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.6, color='teal', label='Predictions')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit')
    plt.title("AI-STEEL: Prediction Accuracy (Strength)")
    plt.xlabel("Actual MPa")
    plt.ylabel("AI Predicted MPa")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "prediction_accuracy.png"))
    
    # Feature Importance Analysis
    plt.figure(figsize=(10, 6))
    sns.barplot(x=features, y=model.feature_importances_, palette="viridis")
    plt.title("Metallurgical Feature Importance")
    plt.ylabel("Importance Score")
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"))
    
    plt.show()
    print(f"Visualizations saved to the /{RESULTS_DIR} folder.")

if __name__ == "__main__":
    final_data = prepare_dataset()
    if not final_data.empty:
        train_and_analyze(final_data)
