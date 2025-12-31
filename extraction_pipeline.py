"""
AI-STEEL: Legacy Metallurgical Extraction & Property Prediction
--------------------------------------------------------------
1. Extract text from PDFs
2. Extract alloy composition, heat treatment parameters & mechanical properties
3. Generate structured CSV dataset
4. Train ML model to predict tensile strength
5. Plot results & export predictions

Author: Kamalu Ikechukwu
"""

import os
import re
import pdfplumber
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

DATA_PATH = "data/"  # 🔥 Change if needed

PDF_FILES = [
    "Old Laboratory Notebook.pdf",
    "Metallurgical Study – AISI Steel Grades.pdf"
]

OUTPUT_CSV = "results/extracted_steel_properties.csv"


# -----------------------------------------------------------
# 1️⃣ PDF TEXT EXTRACTION
# -----------------------------------------------------------
def extract_pdf_text(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# -----------------------------------------------------------
# 2️⃣ SCIENTIFIC ENTITY EXTRACTION (regex-based)
# -----------------------------------------------------------
def extract_entities(text):
    composition = re.findall(r"([A-Z][a-z]?\d{0,2}\s*\d{1,3}\.?\d*%)", text)
    quench_temp = re.findall(r"quench(?:ed)? at (\d{2,4}) ?°?C", text, re.I)
    temper_temp = re.findall(r"temper(?:ed)? at (\d{2,4}) ?°?C", text, re.I)
    tensile_strength = re.findall(r"(\d{3,4})\s?MPa", text, re.I)

    return {
        "composition": ",".join(composition) if composition else None,
        "quenching_temp_C": quench_temp[0] if quench_temp else None,
        "tempering_temp_C": temper_temp[0] if temper_temp else None,
        "tensile_strength_MPa": tensile_strength[0] if tensile_strength else None
    }


# -----------------------------------------------------------
# 3️⃣ PROCESS PDFs → DataFrame
# -----------------------------------------------------------
def process_pdf_files():
    extracted_rows = []
    
    for pdf in PDF_FILES:
        text = extract_pdf_text(DATA_PATH + pdf)
        entities = extract_entities(text)
        
        row = {
            "Source": pdf,
            "Composition": entities["composition"],
            "Quench_Temp_C": entities["quenching_temp_C"],
            "Temper_Temp_C": entities["tempering_temp_C"],
            "Tensile_Strength_MPa": entities["tensile_strength_MPa"]
        }
        extracted_rows.append(row)

    df = pd.DataFrame(extracted_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n✔ Extraction Complete →", OUTPUT_CSV)
    print(df)
    return df


# -----------------------------------------------------------
# 4️⃣ ML MODEL — Predict Strength from Parameters
# -----------------------------------------------------------
def train_ml_model(dataset_path="final_steel_data.csv"):
    df = pd.read_csv(dataset_path)

    df = df.dropna()
    df["Quench_Temp_C"] = df["Quench_Temp_C"].astype(float)
    df["Temper_Temp_C"] = df["Temper_Temp_C"].astype(float)

    X = df[["Quench_Temp_C", "Temper_Temp_C"]]
    y = df["Tensile_Strength_MPa"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n📊 Model Performance")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2 Score:", r2_score(y_test, preds))

    # Plot Predictions
    plt.scatter(y_test, preds)
    plt.xlabel("Actual (MPa)")
    plt.ylabel("Predicted (MPa)")
    plt.title("Tensile Strength Prediction")
    plt.show()

    return model


# -----------------------------------------------------------
# RUNTIME PIPELINE
# -----------------------------------------------------------
if __name__ == "__main__":
    print("\n====== AI-STEEL EXTRACTION & PREDICTION PIPELINE ======\n")

    os.makedirs("results", exist_ok=True)

    extracted_df = process_pdf_files()

    # Load your CSV + PDF extracted CSV together & merge later if you choose
    model = train_ml_model("final_steel_data.csv")

    print("\n🔥 Pipeline Fully Executed — Extraction + ML Completed\n")
