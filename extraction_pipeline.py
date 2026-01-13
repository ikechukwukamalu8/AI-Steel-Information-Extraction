"""
AI-STEEL: Legacy Metallurgical Extraction & Sustainable Property Prediction
Author: Kamalu Ikechukwu
GitHub: https://github.com/ikechukwukamalu8

Description:
An end-to-end pipeline that:
1. Extracts metallurgical data from unstructured PDFs using Regex.
2. Merges extracted data with a structured master CSV dataset.
3. Calculates carbon footprint (CO2) for sustainable materials design.
4. Trains a Random Forest Regressor to predict tensile strength.
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
PDF_FILES = [
    "Old Laboratory Notebook.pdf",
    "Metallurgical Study – AISI Steel Grades.pdf"
]
FINAL_OUTPUT_CSV = os.path.join(RESULTS_DIR, "ai_steel_combined_sustainable.csv")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------
# 1. PDF INFORMATION EXTRACTION
# -----------------------------------------------------------
def extract_pdf_data(filepath):
    """Parses PDF text for quenching, tempering, and strength values."""
    text = ""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    # Extracting numerical values using regular expressions
    # Handles both plain text and basic LaTeX-style formatting
    quench_temp = re.findall(r"quench(?:ed)?\s*(?:at|temperature:)?\s*(\d{3,4})", text, re.I)
    temper_time = re.findall(r"temper(?:ing)?\s*(?:time)?.*?(\d+\.?\d*)\s?hr", text, re.I)
    tensile_strength = re.findall(r"(\d{3,4})\s?MPa", text, re.I)
    
    return {
        "quench_temp": float(quench_temp[0]) if quench_temp else None,
        "temper_time": float(temper_time[0]) if temper_time else None,
        "tensile_strength": float(tensile_strength[0]) if tensile_strength else None,
        "source": os.path.basename(filepath)
    }

# -----------------------------------------------------------
# 2. SUSTAINABILITY CALCULATOR
# -----------------------------------------------------------
def calculate_co2_impact(quench_temp_c, mass_kg=1.0):
    """
    Estimates the CO2 emission (kg) for the heat treatment phase.
    Based on specific heat of steel, furnace efficiency, and grid emission factors.
    """
    if pd.isna(quench_temp_c): return 0
    
    cp_steel = 0.49      # Specific heat (kJ/kg°C)
    t_ambient = 25       # Degrees Celsius
    efficiency = 0.60    # Furnace efficiency factor
    emission_factor = 0.4 # kg CO2 per kWh (Grid average)
    
    # Energy in kWh = (Mass * Cp * DeltaT) / (Efficiency * 3600)
    energy_kwh = (mass_kg * cp_steel * (quench_temp_c - t_ambient)) / (efficiency * 3600)
    return energy_kwh * emission_factor

# -----------------------------------------------------------
# 3. DATA PIPELINE: EXTRACT, MERGE, CLEAN
# -----------------------------------------------------------
def prepare_master_dataset():
    print("Step 1: Extracting data from legacy PDFs...")
    pdf_results = []
    for filename in PDF_FILES:
        path = os.path.join(DATA_DIR, filename)
        data = extract_pdf_data(path)
        if data: pdf_results.append(data)
    
    pdf_df = pd.DataFrame(pdf_results).dropna(subset=['quench_temp'])
    print(f"Extracted {len(pdf_df)} valid samples from documents.")

    print("\nStep 2: Merging with existing CSV database...")
    if os.path.exists(MAIN_DATA_PATH):
        main_df = pd.read_csv(MAIN_DATA_PATH)
        # Combine PDFs and CSV data
        combined_df = pd.concat([main_df, pdf_df], ignore_index=True)
        
        # Add Sustainability Features
        combined_df['carbon_footprint_kgCO2'] = combined_df['quench_temp'].apply(calculate_co2_impact)
        
        combined_df.to_csv(FINAL_OUTPUT_CSV, index=False)
        print(f"Successfully created: {FINAL_OUTPUT_CSV}")
        return combined_df
    else:
        print(f"Error: {MAIN_DATA_PATH} not found in /data directory.")
        return pdf_df

# -----------------------------------------------------------
# 4. MACHINE LEARNING & EVALUATION
# -----------------------------------------------------------
def run_ml_analysis(df):
    print("\nStep 3: Training Predictive Model (Random Forest)...")
    # Focus on clean samples with all necessary ML features
    df_clean = df.dropna(subset=['quench_temp', 'temper_time', 'tensile_strength'])
    
    X = df_clean[["quench_temp", "temper_time"]]
    y = df_clean["tensile_strength"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and Scoring
    y_pred = model.predict(X_test)
    print(f"MAE (Mean Absolute Error): {mean_absolute_error(y_test, y_pred):.2f} MPa")
    print(f"R-Squared Accuracy: {r2_score(y_test, y_pred):.4f}")

    # Visualizing Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Prediction Accuracy Plot
    ax1.scatter(y_test, y_pred, color='#3498db', alpha=0.6)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax1.set_title("Tensile Strength: Actual vs. Predicted")
    ax1.set_xlabel("Measured (MPa)")
    ax1.set_ylabel("AI Predicted (MPa)")
    ax1.grid(True)

    # Sustainability Insight Plot
    sns.regplot(x="quench_temp", y="carbon_footprint_kgCO2", data=df_clean, ax=ax2, color='#27ae60')
    ax2.set_title("Environmental Impact: Temperature vs. CO2")
    ax2.set_xlabel("Quenching Temperature (°C)")
    ax2.set_ylabel("Est. Carbon Footprint (kg CO2/kg steel)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "analysis_plots.png"))
    print(f"Visualizations saved to {RESULTS_DIR}/")
    plt.show()

# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=== AI-STEEL PIPELINE STARTED ===\n")
    final_data = prepare_master_dataset()
    if not final_data.empty:
        run_ml_analysis(final_data)
    print("\n=== PIPELINE EXECUTION COMPLETE ===")
