# 🔥 AI-STEEL: Legacy Metallurgical Extraction & Sustainable Property Prediction

This project implements a multi-modal materials informatics pipeline designed to bridge the gap between legacy metallurgical documentation and modern machine learning. It features a robust extraction engine for unstructured PDFs and a predictive model that integrates chemical composition, heat treatment parameters, and sustainability metrics for material optimization.

## 🔍 Features
- Knowledge Extraction (NLP): Automated digitization of legacy metallurgical records using a SciBERT-ready extraction interface.
- Hybrid Extraction Logic: Utilizes a high-precision, deterministic layer (Regex) to ensure zero-hallucination extraction of critical numerical values (MPa, °C, wt%).
- Sustainable Materials Design: Real-time Carbon Footprint (CO2) estimation based on thermodynamic energy requirements of heat-treatment cycles.
- Domain-Aware Imputation: Intelligent data fusion that fills gaps in chemical compositions using grade-specific statistical averages.
- Predictive Analytics: Random Forest regression for Tensile Strength prediction.
- Advanced Analytics: Generates feature importance rankings and property correlation matrices.
- Explainable AI (XAI): Integrated SHAP (SHapley Additive exPlanations) analysis to decode the nonlinear relationships between alloying elements and mechanical performance, providing a transparent "white-box" view of model decision-making.
  
## 📁 Files
| File | Type | Purpose |
|---|---|---|
| `Old_Laboratory_Notebook.pdf` | PDF | Handwritten notes + heat logs |
| `Study_AISI_Steel_Grades.pdf` | PDF | Alloy composition and strength tables |
| `Study_AISI_Steel_Grades.pdf` | PDF | Alloy composition and strength tables |
| `final_steel_data.csv` | CSV | Clean dataset for model training |

## 🚀 Run Code
```bash
pip install -r requirements.txt
python code/extraction_pipeline.py

