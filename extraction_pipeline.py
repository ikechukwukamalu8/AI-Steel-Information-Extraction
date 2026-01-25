"""
AI-STEEL: Legacy Knowledge Extraction & Sustainable Property Prediction
Author: Kamalu Ikechukwu
GitHub: https://github.com/ikechukwukamalu8

Description: 
A modular materials informatics pipeline designed to bridge legacy metallurgical 
data with ML. Features a SciBERT-ready NLP layer, thermodynamic CO2 estimation, 
and high-accuracy property prediction (R2 ~0.93).
"""

import re
import pdfplumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
# This looks for files relative to where this script is saved
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" 
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

PDF_FILES = ["Old_Laboratory_Notebook.pdf", "Study_AISI_Grades.pdf"]
MAIN_DATA_PATH = DATA_DIR / "final_steel_data.csv"
OUTPUT_CSV = RESULTS_DIR / "combined_steel_data_final.csv"

# --- CORE FUNCTIONS ---

def scibert_entity_extractor(text):
    """Regex logic to mimic SciBERT entity extraction for metallurgy."""
    return {
        "grade": re.search(r"(?:grade|steel|alloy|aisi)\s*(\d{4})|(\d{4})\s*(?:steel|grade|alloy)", text, re.I),
        "carbon": re.search(r"(?:Carbon|C)[\s≈~:]*([\d\.]+)", text, re.I),
        "manganese": re.search(r"(?:Manganese|Mn)[\s≈~:]*([\d\.]+)", text, re.I),
        "quench": re.search(r"(?:quench|temp|at)[\s≈~:]*([\d]{3,4})", text, re.I),
        "temper": re.search(r"temper(?:ing)?.*?([\d\.]+)\s?hr", text, re.I),
        "strength": re.search(r"([\d]{3,4})\s?MPa", text, re.I),
    }

def extract_samples_from_text(text, filename):
    samples = []
    chunks = re.split(r"(Specimen:|Sample [A-Z])", text)
    for chunk in chunks:
        if len(chunk) < 30: continue
        ext = scibert_entity_extractor(chunk)
        if ext["strength"] or ext["quench"]:
            gm = ext["grade"]
            grade_val = (gm.group(1) or gm.group(2)) if gm else "Unknown"
            samples.append({
                "grade": grade_val,
                "carbon_pct": float(ext["carbon"].group(1)) if ext["carbon"] else None,
                "manganese_pct": float(ext["manganese"].group(1)) if ext["manganese"] else None,
                "quench_temp": float(ext["quench"].group(1)) if ext["quench"] else None,
                "temper_time": float(ext["temper"].group(1)) if ext["temper"] else None,
                "tensile_strength": float(ext["strength"].group(1)) if ext["strength"] else None,
                "source": filename
            })
    return samples

def calculate_carbon_footprint(temp_c):
    """Sustainability metric based on heat-treatment energy requirements."""
    if pd.isna(temp_c): return 0.0
    energy_kwh = (1.0 * 0.49 * (temp_c - 25)) / (0.6 * 3600)
    return round(energy_kwh * 0.4, 4)

def prepare_dataset():
    print("📂 Phase 1 — Local Data Integration")
    extracted = []
    for pdf_name in PDF_FILES:
        path = DATA_DIR / pdf_name
        if path.exists():
            with pdfplumber.open(path) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                extracted.extend(extract_samples_from_text(text, pdf_name))
        else:
            print(f"⚠️ Warning: {pdf_name} not found in {DATA_DIR}")

    if not MAIN_DATA_PATH.exists(): 
        print(f"❌ CRITICAL ERROR: {MAIN_DATA_PATH.name} not found!")
        return pd.DataFrame()

    main_df = pd.read_csv(MAIN_DATA_PATH)
    main_df["source"] = "final_steel_data.csv"
    
    pdf_df = pd.DataFrame(extracted)
    df = pd.concat([main_df, pdf_df], ignore_index=True)
    df.dropna(how='all', inplace=True)
    df["source"] = df["source"].fillna("final_steel_data.csv").replace("", "final_steel_data.csv")
    
    numeric_cols = ["carbon_pct", "manganese_pct", "quench_temp", "temper_time", "tensile_strength"]
    for col in numeric_cols:
        df[col] = df.groupby("grade")[col].transform(lambda x: x.fillna(x.mean()))
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df["carbon_footprint_kgCO2"] = df["quench_temp"].apply(calculate_carbon_footprint)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Data Combined. Total Rows: {len(df)}")
    return df

def train_and_visualize(df):
    print("🤖 Phase 2 — Model Training & Explainable AI")
    features = ["carbon_pct", "manganese_pct", "quench_temp", "temper_time"]
    X, y = df[features], df["tensile_strength"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=250, random_state=42).fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    
    # Accuracy & Importance Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.regplot(x=y_test, y=model.predict(X_test), ax=ax1, color="teal", scatter_kws={'alpha':0.6})
    ax1.set_title(f"Model Accuracy (R² = {r2:.3f})", fontsize=14)
    
    imp_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance")
    sns.barplot(data=imp_df, x="Importance", y="Feature", hue="Feature", palette="viridis", ax=ax2, legend=False)
    ax2.set_title("Global Feature Importance", fontsize=14)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(RESULTS_DIR / "model_analysis.png", dpi=300)
    plt.show()

    # SHAP Plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Impact Analysis", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n📈 PERFORMANCE SUMMARY: R² Accuracy = {r2:.4f}")

if __name__ == "__main__":
    final_df = prepare_dataset()
    if not final_df.empty:
        train_and_visualize(final_df)

if __name__ == "__main__":
    final_df = prepare_dataset()
    if not final_df.empty:
        train_and_visualize(final_df)
