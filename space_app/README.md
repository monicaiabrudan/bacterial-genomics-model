---
title: Bacterial 16S Species Classifier
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
short_description: Real-time bacterial species predictor from 16S rRNA DNA sequences.
---

# Bacterial 16S rRNA Species Predictor

This Hugging Face Space hosts an interactive web application that classifies bacterial species from 16S rRNA gene sequences using a fine-tuned **DNA BERT Transformer** model ([mabrudan/bacterial-16s-classifier](https://huggingface.co/mabrudan/bacterial-16s-classifier)).

## Features
- **Instant DNA Analysis**: Predicts species across *Escherichia coli*, *Bacillus subtilis*, *Staphylococcus aureus*, *Pseudomonas aeruginosa*, and *Salmonella enterica*.
- **Confidence Visualization**: Real-time bar chart of class probabilities.
- **K-mer Tokenization**: Powered by overlapping 3-mer genomic tokenization.
