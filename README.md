# 🔥 AI-STEEL: Legacy Metallurgical Extraction & Sustainable Property Prediction

This project builds a pipeline to extract alloy composition, quenching/temper parameters, microstructure notes, and mechanical properties from metallurgical PDFs, then trains Machine Learning models to predict tensile strength.

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
