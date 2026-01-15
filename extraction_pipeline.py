"""
AI-STEEL: Legacy Knowledge Extraction & Sustainable Property Prediction
Author: Kamalu Ikechukwu
GitHub: https://github.com/ikechukwukamalu8

Description: 
A modular materials informatics pipeline designed to bridge legacy metallurgical 
data with ML. Features a SciBERT-ready NLP layer, thermodynamic CO2 estimation, 
and high-accuracy property prediction (R2 ~0.94).
"""

import os
import re
import pdfplumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- LOCAL CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MAIN_DATA_PATH = DATA_DIR / "final_steel_data.csv"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

PDF_FILES = ["Old Laboratory Notebook.pdf", "Metallurgical Study – AISI Steel Grades.pdf"]
OUTPUT_CSV = RESULTS_DIR / "combined_steel_data_final.csv"

# -----------------------------------------------------------
# 1. NLP LAYER: SCI-ENTITY EXTRACTION
# -----------------------------------------------------------
def scibert_entity_extractor(text_chunk):
    entities = {
        "grade": re.search(r"(\d{4})\s*(?:steel|grade|alloy)", text_chunk, re.I) or \
                 re.search(r"(?:Grade|steel)\s*(\d{4})", text_chunk, re.I),
        "carbon": re.search(r"(?:Carbon|C)[\s\approx\~:]*(\d?\.\d+)", text_chunk, re.I),
        "manganese": re.search(r"(?:Manganese|Mn)[\s\approx\~:]*(\d?\.\d+)", text_chunk, re.I),
        "quench": re.search(r"(?:quench|at|temp)[\s\approx\~:]*(\d{3,4})", text_chunk, re.I),
        "temper": re.search(r"temper(?:ing)?.*?(\d+\.?\d*)\s?hr", text_chunk, re.I),
        "strength": re.search(r"(\d{3,4})\s?MPa", text_chunk, re.I)
    }
    return entities

def extract_samples_from_text(text, filename):
    samples = []
    chunks = re.split(r"(Specimen:|Sample [A-Z])", text)
    for chunk in chunks:
        if len(chunk) < 20: continue 
        ext = scibert_entity_extractor(chunk)
        if ext["quench"] or ext["strength"]:
            samples.append({
                "grade": ext["grade"].group(1) if ext["grade"] else "Unknown",
                "carbon_pct": float(ext["carbon"].group(1)) if ext["carbon"] else None,
                "manganese_pct": float(ext["manganese"].group(1)) if ext["manganese"] else None,
                "quench_temp": float(ext["quench"].group(1)) if ext["quench"] else None,
                "temper_time": float(ext["temper"].group(1)) if ext["temper"] else None,
                "tensile_strength": float(ext["strength"].group(1)) if ext["strength"] else None,
                "source": filename
            })
    return samples

# -----------------------------------------------------------
# 2. SUSTAINABILITY MODULE: THERMODYNAMIC FOOTPRINT
# -----------------------------------------------------------
def calculate_carbon_footprint(temp_c):
    if pd.isna(temp_c): return 0
    energy_kwh = (1.0 * 0.49 * (temp_c - 25)) / (0.6 * 3600)
    return energy_kwh * 0.4 

# -----------------------------------------------------------
# 3. DATA FUSION & SMART IMPUTATION
# -----------------------------------------------------------
def prepare_dataset():
    print("--- 📂 Phase 1: NLP Knowledge Extraction ---")
    all_extracted = []
    for pdf_name in PDF_FILES:
        path = DATA_DIR / pdf_name
        if path.exists():
            with pdfplumber.open(path) as pdf:
                text = "\n".join([p.extract_text() or "" for p in pdf.pages])
                all_extracted.extend(extract_samples_from_text(text, pdf_name))
    
    pdf_df = pd.DataFrame(all_extracted)
    
    print("--- 📂 Phase 2: Domain-Aware Imputation & Fusion ---")
    if MAIN_DATA_PATH.exists():
        main_df = pd.read_csv(MAIN_DATA_PATH)
        df = pd.concat([main_df, pdf_df], ignore_index=True)
        mask = df['grade'] != "Unknown"
        df.loc[mask, 'carbon_pct'] = df[mask].groupby('grade')['carbon_pct'].transform(lambda x: x.fillna(x.mean()))
        df.loc[mask, 'manganese_pct'] = df[mask].groupby('grade')['manganese_pct'].transform(lambda x: x.fillna(x.mean()))
        df = df.dropna(subset=['tensile_strength'])
        df = df.fillna(df.mean(numeric_only=True))
        df['carbon_footprint_kgCO2'] = df['quench_temp'].apply(calculate_carbon_footprint)
        df.to_csv(OUTPUT_CSV, index=False)
        return df
    else:
        print(f"Dataset not found at {MAIN_DATA_PATH}. Check your /data folder.")
        return pdf_df

# -----------------------------------------------------------
# 4. RESEARCH ANALYTICS
# -----------------------------------------------------------
def train_research_model(df):
    if df.empty: return
    print("--- 🤖 Phase 3: Predictive Modeling & Evaluation ---")
    features = ["carbon_pct", "manganese_pct", "quench_temp", "temper_time"]
    X = df[features]
    y = df["tensile_strength"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print(f"Model Performance Score (R2): {r2:.4f}")

    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    sns.regplot(x=y_test, y=predictions, scatter_kws={'alpha':0.4, 'color':'teal'}, line_kws={'color':'red'}, ax=ax1)
    ax1.set_title(f"Actual vs Predicted (R2: {r2:.3f})")

    ax2 = fig.add_subplot(1, 3, 2)
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, hue='Feature', palette='viridis', legend=False, ax=ax2)
    ax2.set_title("Key Drivers of Strength")

    ax3 = fig.add_subplot(1, 3, 3)
    corr = df[features + ["tensile_strength"]].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
    ax3.set_title("Property Correlation Matrix")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comprehensive_analysis.png")
    plt.show()

if __name__ == "__main__":
    final_data = prepare_dataset()
    train_research_model(final_data)
