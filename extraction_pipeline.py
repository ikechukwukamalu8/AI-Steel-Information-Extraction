"""
AI-STEEL: Legacy Knowledge Extraction & Sustainable Design
Author: Kamalu Ikechukwu
Description: A pipeline to digitize legacy metallurgical notes and 
             predict tensile strength with sustainability metrics.
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
# GitHub standard: Put data in a folder, results in another
DATA_DIR = "data"
RESULTS_DIR = "results"
MAIN_DATA_PATH = os.path.join(DATA_DIR, "final_steel_data.csv")
PDF_FILES = ["Old Laboratory Notebook.pdf", "Metallurgical Study – AISI Steel Grades.pdf"]
OUTPUT_CSV = os.path.join(RESULTS_DIR, "combined_steel_data_final.csv")

# Create results folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------
# 1. SCIENTIFIC KNOWLEDGE EXTRACTION
# -----------------------------------------------------------
def extract_samples_from_text(text, filename):
    samples = []
    # Logic to handle multiple specimens in one document
    chunks = re.split(r"(Specimen:|Sample [A-Z])", text)
    
    for chunk in chunks:
        if len(chunk) < 20: continue 
        
        # Regex to find Carbon, Manganese, Quench, and Strength
        # Optimized for LaTeX symbols like \approx and ^\circ
        grade = re.search(r"(?:Grade|steel)\s*(\d{4})", chunk, re.I)
        carbon = re.search(r"(?:Carbon|C)[\s\approx\~:]*(\d?\.\d+)", chunk, re.I)
        manganese = re.search(r"(?:Manganese|Mn)[\s\approx\~:]*(\d?\.\d+)", chunk, re.I)
        quench = re.search(r"(?:quench|at|temperature:)[\s\approx\~]*(\d{3,4})", chunk, re.I)
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
# 2. DATA PIPELINE (Merge & Impute)
# -----------------------------------------------------------
def prepare_dataset():
    print("--- 📂 AI-STEEL: Extracting Data ---")
    all_extracted = []
    
    for pdf_name in PDF_FILES:
        path = os.path.join(DATA_DIR, pdf_name)
        if os.path.exists(path):
            with pdfplumber.open(path) as pdf:
                text = "\n".join([p.extract_text() or "" for p in pdf.pages])
                all_extracted.extend(extract_samples_from_text(text, pdf_name))
    
    pdf_df = pd.DataFrame(all_extracted)
    
    # Ensure standard columns exist
    cols = ['grade', 'carbon_pct', 'manganese_pct', 'quench_temp', 'temper_time', 'tensile_strength']
    for c in cols:
        if c not in pdf_df.columns: pdf_df[c] = np.nan

    print("--- 📂 AI-STEEL: Merging & Filling Gaps ---")
    if os.path.exists(MAIN_DATA_PATH):
        main_df = pd.read_csv(MAIN_DATA_PATH)
        combined_df = pd.concat([main_df, pdf_df], ignore_index=True)
        
        # FIX: Fill missing Carbon/Mn based on the average for that Steel Grade
        combined_df['carbon_pct'] = combined_df.groupby('grade')['carbon_pct'].transform(lambda x: x.fillna(x.mean()))
        combined_df['manganese_pct'] = combined_df.groupby('grade')['manganese_pct'].transform(lambda x: x.fillna(x.mean()))
        
        # Fill remaining gaps with global averages
        combined_df = combined_df.fillna(combined_df.mean(numeric_only=True))
        
        combined_df.to_csv(OUTPUT_CSV, index=False)
        return combined_df
    return pdf_df

# -----------------------------------------------------------
# 3. ML MODEL & FEATURE IMPORTANCE
# -----------------------------------------------------------
def train_model(df):
    print("--- 🤖 AI-STEEL: Training Model ---")
    features = ["carbon_pct", "manganese_pct", "quench_temp", "temper_time"]
    df_ml = df.dropna(subset=features + ["tensile_strength"])
    
    X = df_ml[features]
    y = df_ml["tensile_strength"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Final R2 Accuracy: {model.score(X_test, y_test):.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, model.predict(X_test), alpha=0.6, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.title("Steel Property Prediction (Chemistry + Heat Treatment)")
    plt.xlabel("Actual MPa")
    plt.ylabel("AI Predicted MPa")
    
    # Save the plot for GitHub display
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_plot.png"))
    plt.show()

if __name__ == "__main__":
    final_data = prepare_dataset()
    if not final_data.empty:
        train_model(final_data)
