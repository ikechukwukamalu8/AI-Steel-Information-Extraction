# 🔥 AI-STEEL: Legacy Metallurgical Extraction & Sustainable Property Prediction

This project is a multi-modal pipeline designed to bridge the gap between legacy metallurgical knowledge and modern machine learning. It features a robust extraction engine for unstructured PDFs and a predictive model that integrates chemical composition and heat treatment for sustainable material optimization.

## 🔍 Features
- PDF text + table extraction
- Scientific entity extraction using spaCy + regex
- Structured dataset generation (.csv)
- ML regression model for tensile strength prediction
- Expandable to SciBERT/MatsciBERT entity recognition


## 🔍 Features
- PDF text + table extraction
- Scientific entity extraction using spaCy + regex
- Structured dataset generation (.csv)
- ML regression model for tensile strength prediction
- Expandable to SciBERT/MatsciBERT entity recognition

## 📁 Files
| File | Type | Purpose |
|---|---|---|
| `Old Laboratory Notebook.pdf` | PDF | Handwritten notes + heat logs |
| `Metallurgical Study – AISI Steel Grades.pdf` | PDF | Alloy composition and strength tables |
| `final_steel_data.csv` | CSV | Clean dataset for model training |

## 🚀 Run Code
```bash
pip install -r requirements.txt
python code/extraction_pipeline.py

Future Work

Fine-tune SciBERT for domain extraction

Build a multimodal property prediction model

Benchmark against baselines for PhD-grade publication
