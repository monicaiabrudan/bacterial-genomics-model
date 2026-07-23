"""
Train a DNA Sequence Classifier on Real NCBI Bacterial Sequences
Fine-tunes a genomic sequence classification model to predict bacterial species.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

DATA_PATH = Path(__file__).parent / "data" / "bacterial_16S_sequences.json"
OUTPUT_DIR = Path(__file__).parent / "model_output"

# Mapping species labels
ID2LABEL = {
    0: "Escherichia coli",
    1: "Bacillus subtilis",
    2: "Staphylococcus aureus",
    3: "Pseudomonas aeruginosa",
    4: "Salmonella enterica"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# K-mer DNA Tokenizer helper (overlapping 3-mers)
class DNATokenizer:
    """Overlapping 3-mer genomic tokenizer for bacterial sequence classification."""
    def __init__(self, k: int = 3):
        self.k = k
        bases = ["A", "C", "G", "T"]
        import itertools
        kmers = ["".join(p) for p in itertools.product(bases, repeat=k)]
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        for idx, kmer in enumerate(kmers, start=4):
            self.vocab[kmer] = idx
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.cls_token_id = 2
        self.sep_token_id = 3

    def seq_to_kmers(self, seq: str) -> list[str]:
        seq = "".join([c for c in seq.upper() if c in "ACGT"])
        return [seq[i : i + self.k] for i in range(len(seq) - self.k + 1)]

    def __call__(self, sequences, max_length=512, truncation=True, padding=True, return_tensors=None):
        if isinstance(sequences, str):
            sequences = [sequences]
            
        input_ids = []
        attention_mask = []
        
        for seq in sequences:
            kmers = self.seq_to_kmers(seq)[: max_length - 2]
            ids = [self.cls_token_id] + [self.vocab.get(km, 1) for km in kmers] + [self.sep_token_id]
            mask = [1] * len(ids)
            input_ids.append(ids)
            attention_mask.append(mask)
            
        # Pad batch
        if padding:
            max_batch_len = max(len(i) for i in input_ids)
            for i in range(len(input_ids)):
                pad_len = max_batch_len - len(input_ids[i])
                input_ids[i] += [self.pad_token_id] * pad_len
                attention_mask[i] += [0] * pad_len
                
        res = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_tensors == "pt":
            res = {k: torch.tensor(v) for k, v in res.items()}
        return res

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Run download_real_bacterial_data.py first.")
        
    print("Loading real bacterial sequence dataset...")
    with open(DATA_PATH, "r") as f:
        data_list = json.load(f)
        
    print(f"Loaded {len(data_list)} real bacterial DNA samples.")
    
    # Create Hugging Face Dataset
    raw_dataset = Dataset.from_list(data_list)
    dataset_dict = raw_dataset.train_test_split(test_size=0.2, seed=42)
    
    tokenizer = DNATokenizer(k=3)
    
    def tokenize_batch(batch):
        return tokenizer(batch["sequence"], max_length=512, padding=False)
        
    tokenized_dataset = dataset_dict.map(tokenize_batch, batched=True)
    
    print("Initializing DNA Sequence Transformer Model...")
    # Config for DNA BERT transformer
    config = BertConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id
    )
    
    model = BertForSequenceClassification(config)
    
    # Custom Data Collator for padding
    def data_collator(features):
        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention_mask = [f["attention_mask"] for f in features]
        batch_labels = [f["label"] for f in features]
        
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids, padded_masks = [], []
        for ids, mask in zip(batch_input_ids, batch_attention_mask):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            padded_masks.append(mask + [0] * pad_len)
            
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long)
        }
        
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)
        
    hf_username = os.getenv("HF_USERNAME", "your-username")
    repo_id = f"{hf_username}/bacterial-16S-species-classifier"
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,  # Set to True when ready to push to HF
        hub_model_id=repo_id,
        logging_steps=5,
        save_total_limit=1,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("\nStarting training on real bacterial sequences...")
    trainer.train()
    
    print("\nEvaluating model performance...")
    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy'] * 100:.2f}%")
    
    # Save model locally
    model.save_pretrained(OUTPUT_DIR / "final_model")
    print(f"Model saved to: {OUTPUT_DIR / 'final_model'}")

if __name__ == "__main__":
    main()
