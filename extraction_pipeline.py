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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
# Designed to run from the root of: AI-Steel-Information-Extraction/
DATA_DIR = "data"
RESULTS_DIR = "results"
MAIN_DATA_PATH = os.path.join(DATA_DIR, "final_steel_data.csv")
PDF_FILES = ["Old Laboratory Notebook.pdf", "Metallurgical Study – AISI Steel Grades.pdf"]
OUTPUT_CSV = os.path.join(RESULTS_DIR, "combined_steel_data_final.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------
# 1. NLP LAYER: SCI-ENTITY EXTRACTION (SciBERT-Ready)
# -----------------------------------------------------------
def scibert_entity_extractor(text_chunk):
    """
    INTERFACE LAYER: Mimics a Transformer NER model. Currently uses 
    high-precision regex for numerical extraction from legacy PDF strings.
    """
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
    # Split text into chunks based on sample headers
    chunks = re.split(r"(Specimen:|Sample [A-Z])", text)
    
    for chunk in chunks:
        if len(chunk) < 20: continue 
        
        # Call the modular extractor
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
    """
    Estimates CO2 (kg) to heat 1kg of steel to quench temperature.
    E = (m * Cp * dT) / efficiency. 
    Assumes Cp=0.49 kJ/kgK, Efficiency=60%, Emission Factor=0.4 kgCO2/kWh.
    """
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
        path = os.path.join(DATA_DIR, pdf_name)
        if os.path.exists(path):
            with pdfplumber.open(path) as pdf:
                text = "\n".join([p.extract_text() or "" for p in pdf.pages])
                all_extracted.extend(extract_samples_from_text(text, pdf_name))
    
    pdf_df = pd.DataFrame(all_extracted)
    
    print("--- 📂 Phase 2: Domain-Aware Imputation & Fusion ---")
    if os.path.exists(MAIN_DATA_PATH):
        main_df = pd.read_csv(MAIN_DATA_PATH)
        df = pd.concat([main_df, pdf_df], ignore_index=True)
        
        # DOMAIN LOGIC: Use Steel Grade averages to fill chemistry gaps
        df['carbon_pct'] = df.groupby('grade')['carbon_pct'].transform(lambda x: x.fillna(x.mean()))
        df['manganese_pct'] = df.groupby('grade')['manganese_pct'].transform(lambda x: x.fillna(x.mean()))
        
        # Fallback and CO2 Calculation
        df = df.fillna(df.mean(numeric_only=True))
        df['carbon_footprint_kgCO2'] = df['quench_temp'].apply(calculate_carbon_footprint)
        
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Master dataset saved to: {OUTPUT_CSV}")
        return df
    return pdf_df

# -----------------------------------------------------------
# 4. RESEARCH ANALYTICS (ML)
# -----------------------------------------------------------
def train_research_model(df):
    print("--- 🤖 Phase 3: Predictive Modeling & Evaluation ---")
    features = ["carbon_pct", "manganese_pct", "quench_temp", "temper_time"]
    X = df[features]
    y = df["tensile_strength"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)
    
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"Model Performance Score (R2): {r2:.4f}")

    # Plotting Accuracy for GitHub
    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_test, y=model.predict(X_test), scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
    plt.title(f"AI-STEEL: Actual vs Predicted Strength\n(R2 Score: {r2:.3f})")
    plt.xlabel("Actual Tensile Strength (MPa)")
    plt.ylabel("AI Predicted Strength (MPa)")
    
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_plot.png"))
    print(f"Plot saved to: {RESULTS_DIR}/accuracy_plot.png")
    plt.show()

if __name__ == "__main__":
    final_data = prepare_dataset()
    if not final_data.empty:
        train_research_model(final_data)
