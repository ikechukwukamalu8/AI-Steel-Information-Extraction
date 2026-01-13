"""
AI-STEEL: Legacy Metallurgical Extraction & Sustainable Property Prediction
Author: Kamalu Ikechukwu
Description: 
An end-to-end pipeline that extracts metallurgical data from unstructured PDFs, merges it with structured CSV data, and trains a Random Forest model to predict tensile strength.
"""

import os
import re
import pdfplumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
FINAL_COMBINED_CSV = os.path.join(RESULTS_DIR, "combined_steel_data.csv")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------
# 1. PDF TEXT EXTRACTION
# -----------------------------------------------------------
def extract_pdf_text(filepath):
    """Reads text from a PDF file."""
    text = ""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return ""
    
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# -----------------------------------------------------------
# 2. SCIENTIFIC ENTITY EXTRACTION (Regex)
# -----------------------------------------------------------
def extract_entities(text):
    """
    Extracts quenching temp, tempering time, and tensile strength using regex.
    Patterns are adjusted to handle scientific notation and units.
    """
    # Regex for Quenching Temperature (e.g., 830°C or 830C)
    quench_temp = re.findall(r"quench(?:ed)? at (\d{2,4}) ?°?C", text, re.I)
    
    # Regex for Tempering Time (e.g., 2 hr or 1.5 hr)
    temper_time = re.findall(r"temper(?:ing)?\s*(?:time)?.*?(\d+\.?\d*)\s?hr", text, re.I)
    
    # Regex for Tensile Strength (e.g., 620 MPa)
    tensile_strength = re.findall(r"(\d{3,4})\s?MPa", text, re.I)
    
    return {
        "quench_temp": float(quench_temp[0]) if quench_temp else None,
        "temper_time": float(temper_time[0]) if temper_time else None,
        "tensile_strength": float(tensile_strength[0]) if tensile_strength else None
    }

# -----------------------------------------------------------
# 3. DATA PROCESSING & MERGING
# -----------------------------------------------------------
def prepare_dataset():
    """Extracts data from PDFs and merges it with the existing CSV dataset."""
    print("--- Starting Data Extraction & Merging ---")
    
    # Extract from PDFs
    extracted_rows = []
    for pdf_name in PDF_FILES:
        path = os.path.join(DATA_DIR, pdf_name)
        text = extract_pdf_text(path)
        entities = extract_entities(text)
        entities['source'] = pdf_name
        extracted_rows.append(entities)
    
    pdf_df = pd.DataFrame(extracted_rows)
    pdf_df = pdf_df.dropna(subset=['quench_temp', 'temper_time', 'tensile_strength'])
    
    print(f"Extracted {len(pdf_df)} valid samples from legacy PDFs.")

    # Load Main CSV
    if os.path.exists(MAIN_DATA_PATH):
        main_df = pd.read_csv(MAIN_DATA_PATH)
        # Ensure column names match for merging
        # Current main_df columns: grade, carbon_pct, manganese_pct, quench_temp, temper_time, tensile_strength
        combined_df = pd.concat([main_df, pdf_df], ignore_index=True)
    else:
        print(f"Warning: {MAIN_DATA_PATH} not found. Using PDF data only.")
        combined_df = pdf_df

    # Save final dataset
    combined_df.to_csv(FINAL_COMBINED_CSV, index=False)
    print(f"Combined dataset saved to: {FINAL_COMBINED_CSV}")
    return combined_df

# -----------------------------------------------------------
# 4. MACHINE LEARNING MODEL
# -----------------------------------------------------------
def train_and_evaluate(df):
    """Trains a Random Forest Regressor and plots performance."""
    print("\n--- Training Machine Learning Model ---")
    
    # Clean data for ML
    ml_data = df.dropna(subset=['quench_temp', 'temper_time', 'tensile_strength'])
    
    X = ml_data[["quench_temp", "temper_time"]]
    y = ml_data["tensile_strength"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Performance:")
    print(f"Mean Absolute Error: {mae:.2f} MPa")
    print(f"R2 Score: {r2:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, label="Predicted vs Actual")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label="Perfect Prediction")
    plt.xlabel("Actual Tensile Strength (MPa)")
    plt.ylabel("Predicted Tensile Strength (MPa)")
    plt.title("AI-STEEL: Prediction Accuracy")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(RESULTS_DIR, "prediction_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.show()

# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("   AI-STEEL RESEARCH PIPELINE INITIALIZED")
    print("==============================================\n")
    
    # 1. Prepare/Merge Data
    combined_data = prepare_dataset()
    
    # 2. Run ML Pipeline
    train_and_evaluate(combined_data)
    
    print("\nPipeline Execution Complete.")
