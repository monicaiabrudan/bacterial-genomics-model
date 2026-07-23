"""
Inference Script: Predict Bacterial Species from a DNA Sequence
Loads the trained genomic sequence classifier model and predicts the species.
"""

import sys
import torch
from pathlib import Path
from transformers import BertForSequenceClassification

MODEL_DIR = Path(__file__).parent / "model_output" / "final_model"

# K-mer DNA Tokenizer helper (overlapping 3-mers)
class DNATokenizer:
    def __init__(self, k: int = 3):
        self.k = k
        bases = ["A", "C", "G", "T"]
        import itertools
        kmers = ["".join(p) for p in itertools.product(bases, repeat=k)]
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        for idx, kmer in enumerate(kmers, start=4):
            self.vocab[kmer] = idx
        self.pad_token_id = 0
        self.cls_token_id = 2
        self.sep_token_id = 3

    def seq_to_kmers(self, seq: str) -> list[str]:
        seq = "".join([c for c in seq.upper() if c in "ACGT"])
        return [seq[i : i + self.k] for i in range(len(seq) - self.k + 1)]

    def __call__(self, sequence, max_length=512):
        kmers = self.seq_to_kmers(sequence)[: max_length - 2]
        ids = [self.cls_token_id] + [self.vocab.get(km, 1) for km in kmers] + [self.sep_token_id]
        mask = [1] * len(ids)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long)
        }

def predict_dna_sequence(dna_sequence: str):
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Trained model directory not found at {MODEL_DIR}. Please run train_real_bacterial_model.py first.")
        
    tokenizer = DNATokenizer()
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    
    inputs = tokenizer(dna_sequence)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        
    id2label = model.config.id2label
    predicted_species = id2label[pred_id]
    confidence = probs[pred_id].item() * 100
    
    print(f"\n--- Genomic Prediction Results ---")
    print(f"Input DNA Length: {len(dna_sequence)} bp")
    print(f"Predicted Species: {predicted_species}")
    print(f"Confidence Score:  {confidence:.2f}%\n")
    print("Class Probabilities:")
    for label_id, prob in enumerate(probs):
        species_name = id2label.get(label_id, str(label_id))
        print(f"  - {species_name:25s}: {prob.item()*100:6.2f}%")
        
    return predicted_species, confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample_seq = sys.argv[1]
    else:
        # Sample E. coli 16S rRNA sequence segment
        sample_seq = "AGAGTTTGATCATGGCTCAGATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACGGTAACAGGAAGAAGCTTGCTTCTTTGCTGACGAGTGGCGGACGGGTGAGTAATGTCTGGGAAACTGCCTGATGGAGGGGGATAACTACTGGAAACGGTAGCTAATACCGCATAACGTCGCAAGACCAAAGAGGGGGACCTTCGGGCCTCTTGCCATCGGATGTGCCCAGATGGGATTAGCTAGTAGGTGGGGTAACGGCTCACCTAGGCGACGATCCCTAGCTGGTCTGAGAGGATGACCAGCCACACTGGAACTGAGACACGGTCCAGACTCCTACGGGAGGCAGCAGTGGGGAATATTGCACAATGGGCGCAAGCCTGATGCAGCCATGCCGCGTGTATGAAGAAGGCCTTCGGGTTGTAAAGTACTTTCAGCGGGGAGGAAGGGAGTAAAGTTAATACCTTTGCTCATTGACGTTACCCGCAGAAGAAGCACCGGCTAACTCCGTGCCAGCAGCCGCGGTAATACGGAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGCACGCAGGCGGTTTGTTAAGTCAGATGTGAAATCCCCGGGCTCAACCTGGGAACTGCATCTGATACTGGCAAGCTTGAGTCTCGTAGAGGGGGGTAGAATTCCAGGTGTAGCGGTGAAATGCGTAGAGATCTGGAGGAATACCGGTGGCGAAGGCGGCCCCCTGGACGAAGACTACGCTCAGGTGCGAAAGCGTGGGGAGCAAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGATGTCGACTTGGAGGTTGTGCCCTTGAGGCGTGGCTTCCGGAGCTAACGCGTTAAGTCGACCGCCTGGGGAGTACGGCCGCAAGGTTAAAACTCAAATGAATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGATGCAACGCGAAGAACCTTACCTGGTCTTGACATCCACGGAAGTTTTCAGAGATGAGAATGTGCCTTCGGGAACCGTGAGACAGGTGCTGCATGGCTGTCGTCAGCTCGTGTTGTGAAATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTATCCTTTGTTGCCAGCGGTCCGGCCGGGAACTCAAAGGAGACTGCCAGTGATAAACTGGAGGAAGGTGGGGATGACGTCAAGTCATCATGGCCCTTACGACCAGGGCTACACACGTGCTACAATGGCGCATACAAAGAGAAGCGACCTCGCGAGAGCAAGCGGACCTCATAAAGTGCGTCGTAGTCCGGATTGGAGTCTGCAACTCGACTCCATGAAGTCGGAATCGCTAGTAATCGTGGATCAGAATGCCACGGTGAATACGTTCCCGGGCCTTGTACACACCGCCGTCACACCATGGGAGTGGGTTGCAAAAGAAGTAGGTAGCTTAACCTTCGGGAGGGCGCTTACCACTTTGTGATTCATGACTGGGGTGAAGTCGTAACAAGGTAACCGTAGGGGAACCTGCGGTTGGATCACCTCCTT"
        
    predict_dna_sequence(sample_seq)
