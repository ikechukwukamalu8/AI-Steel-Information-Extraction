# 🔥 AI-STEEL: Legacy Metallurgical Extraction & Sustainable Property Prediction

This project demonstrates the extraction of scientific knowledge from legacy PDF documents and integrates it into a sustainable ML framework. It estimates the carbon intensity of manufacturing processes alongside mechanical property prediction, enabling multi-objective material design.

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
