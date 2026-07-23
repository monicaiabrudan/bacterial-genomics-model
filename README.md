# Bacterial Genomics Model with Hugging Face

This project provides a complete end-to-end template to build, fine-tune, evaluate, and publish a DNA sequence classification model for bacterial genomics on [Hugging Face](https://huggingface.co/).

---

## 📁 Folder Structure

```
bacterial_genomics_hf/
├── README.md                      # Complete guide and documentation
├── requirements.txt               # Required Python packages
└── bacterial_genomics_hf_demo.py  # Ready-to-run Python training & publishing script
```

---

## 🚀 Quick Start

### 1. Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Hugging Face Authentication
Log in to your Hugging Face account via CLI (get a write token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens)):
```bash
huggingface-cli login
```

### 3. Run the Training & Publishing Script
Run the sample fine-tuning pipeline:
```bash
python bacterial_genomics_hf_demo.py
```

---

## 📖 Step-by-Step Guide

### Selecting a Pre-trained Genomic Model
Rather than training from scratch, fine-tuning pre-trained foundation models yields strong performance on DNA tasks:

- **`InstaDeepAI/nucleotide-transformer-500m-human-ref`**: Nucleotide Transformer fine-tuned for sequence prediction.
- **`zhihan1996/DNABERT-2-117M`**: DNABERT-2 architecture suited for genomic feature classification.

### Workflow Summary

1. **Prepare Data**: Convert FASTA or CSV sequence files into Hugging Face `Dataset` format (`sequence` string + `label` integer).
2. **Tokenization**: Tokenize DNA sequences using the genomic model's tokenizer.
3. **Training & Evaluation**: Use Hugging Face `Trainer` to fine-tune the classification head.
4. **Publishing (`push_to_hub`)**: Upload model weights, tokenizer configs, and evaluation metrics directly to Hugging Face Hub.

---

## 📄 Model Card Template (`README.md` for Hugging Face)

When publishing your model on Hugging Face, include this model card structure in your repository:

```markdown
---
language: dna
license: mit
tags:
- genomics
- biology
- bioinformatics
- bacteria
- sequence-classification
metrics:
- accuracy
---

# Bacterial DNA Sequence Classifier

## Model Description
This model classifies bacterial DNA sequences (e.g., promoter sites, taxonomic families, or antimicrobial resistance genes).

## Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "your-username/bacterial-promoter-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sequence = "ATGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1).item()
print("Prediction:", prediction)
```
```
