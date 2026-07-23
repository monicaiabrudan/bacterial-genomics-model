import itertools
import torch
import gradio as gr
from transformers import BertForSequenceClassification

# K-mer DNA Tokenizer helper (overlapping 3-mers)
class DNATokenizer:
    def __init__(self, k: int = 3):
        self.k = k
        bases = ["A", "C", "G", "T"]
        kmers = ["".join(p) for p in itertools.product(bases, repeat=k)]
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        for idx, kmer in enumerate(kmers, start=4):
            self.vocab[kmer] = idx
        self.cls_token_id = 2
        self.sep_token_id = 3

    def seq_to_kmers(self, seq: str) -> list[str]:
        seq = "".join([c for c in seq.upper() if c in "ACGT"])
        return [seq[i : i + self.k] for i in range(len(seq) - self.k + 1)]

    def __call__(self, sequence: str, max_length=512):
        kmers = self.seq_to_kmers(sequence)[: max_length - 2]
        ids = [self.cls_token_id] + [self.vocab.get(km, 1) for km in kmers] + [self.sep_token_id]
        mask = [1] * len(ids)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long)
        }

MODEL_ID = "mabrudan/bacterial-16s-classifier"
print(f"Loading model from Hugging Face Hub ({MODEL_ID})...")
model = BertForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
tokenizer = DNATokenizer(k=3)

# Default sample sequences for quick testing
SAMPLE_ECOLI = "AGAGTTTGATCATGGCTCAGATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACGGTAACAGGAAGAAGCTTGCTTCTTTGCTGACGAGTGGCGGACGGGTGAGTAATGTCTGGGAAACTGCCTGATGGAGGGGGATAACTACTGGAAACGGTAGCTAATACCGCATAACGTCGCAAGACCAAAGAGGGGGACCTTCGGGCCTCTTGCCATCGGATGTGCCCAGATGGGATTAGCTAGTAGGTGGGGTAACGGCTCACCTAGGCGACGATCCCTAGCTGGTCTGAGAGGATGACCAGCCACACTGGAACTGAGACACGGTCCAGACTCCTACGGGAGGCAGCAGTGGGGAATATTGCACAATGGGCGCAAGCCTGATGCAGCCATGCCGCGTGTATGAAGAAGGCCTTCGGGTTGTAAAGTACTTTCAGCGGGGAGGAAGGGAGTAAAGTTAATACCTTTGCTCATTGACGTTACCCGCAGAAGAAGCACCGGCTAACTCCGTGCCAGCAGCCGCGGTAATACGGAGGGTGCAAGCGTTAATCGGAATTACTGGGCGTAAAGCGCACGCAGGCGGTTTGTTAAGTCAGATGTGAAATCCCCGGGCTCAACCTGGGAACTGCATCTGATACTGGCAAGCTTGAGTCTCGTAGAGGGGGGTAGAATTCCAGGTGTAGCGGTGAAATGCGTAGAGATCTGGAGGAATACCGGTGGCGAAGGCGGCCCCCTGGACGAAGACTACGCTCAGGTGCGAAAGCGTGGGGAGCAAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGATGTCGACTTGGAGGTTGTGCCCTTGAGGCGTGGCTTCCGGAGCTAACGCGTTAAGTCGACCGCCTGGGGAGTACGGCCGCAAGGTTAAAACTCAAATGAATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGATGCAACGCGAAGAACCTTACCTGGTCTTGACATCCACGGAAGTTTTCAGAGATGAGAATGTGCCTTCGGGAACCGTGAGACAGGTGCTGCATGGCTGTCGTCAGCTCGTGTTGTGAAATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTATCCTTTGTTGCCAGCGGTCCGGCCGGGAACTCAAAGGAGACTGCCAGTGATAAACTGGAGGAAGGTGGGGATGACGTCAAGTCATCATGGCCCTTACGACCAGGGCTACACACGTGCTACAATGGCGCATACAAAGAGAAGCGACCTCGCGAGAGCAAGCGGACCTCATAAAGTGCGTCGTAGTCCGGATTGGAGTCTGCAACTCGACTCCATGAAGTCGGAATCGCTAGTAATCGTGGATCAGAATGCCACGGTGAATACGTTCCCGGGCCTTGTACACACCGCCGTCACACCATGGGAGTGGGTTGCAAAAGAAGTAGGTAGCTTAACCTTCGGGAGGGCGCTTACCACTTTGTGATTCATGACTGGGGTGAAGTCGTAACAAGGTAACCGTAGGGGAACCTGCGGTTGGATCACCTCCTT"
SAMPLE_SAUREUS = "CGCGGCACCTACCATGCAGTCGAGCGAACGGACGAGAAGCTTGCTTCTCTGATGTTAGCGGCGGACGGGTGAGTAACACGTGGGTAACCTACCTATAAGACTGGGATAACTCCGGGAAACCGGGGCTAATACCGGATAACATTTTGAACCGCATGGTTCAAAAGTGAAAGACGGTCTTGCTGTCACTTATAGATGGACCCGCGGCGCATTAGCTAGTTGGTAAGGTAACGGCTTACCAAGGCGACGATGCGTAGCCGACCTGAGAGGGTGATCGGCCACACTGGAACTGAGACACGGTCCAGACTCCTACGGGAGGCAGCAGTAGGGAATCTTCCGCAATGGGCGAAAGCCTGACGGAGCAACGCCGCGTGAGTGATGAAGGTCTTCGGATCGTAAAACTCTGTTATTAGGGAAGAACAAATGTGTAAGTAACTATGCACATCTTGACGGTACCTAATCAGAAAGCCACGGCTAACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCAAGCGTTATCCGGAATTATTGGGCGTAAAGCGCGCGTAGGCGGTTTCTTAAGTCTGATGTGAAAGCCCACGGCTCAACCGTGGAGGGTCATTGGAAACTGGGGAACTTGAGTGCAGAAGAGGAAAGTGGAATTCCATGTGTAGCGGTGAAATGCGCAGAGATATGGAGGAACACCAGTGGCGAAGGCGACTTTCTGGTCTGTAACTACGCTGAGGCGCGAAAGCGTGGGGAGCAAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGATGAGTGCTAAGTGTTAGGGGGTTTCCGCCCCTTAGTGCTGCAGCTAACGCATTAAGCACTCCGCCTGGGGAGTACGACCGCAAGGTTGAAACTCAAAGGAATTGACGGGGACCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGAAGCAACGCGAAGAACCTTACCAAATCTTGACATCCTTTGACAACTCTAGAGATAGAGCCTTCCCCTTCGGGGGACAAAGTGACAGGTGCTGCATGGCTGTCGTCAGCTCGTGTTGTGAAATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTAAGCTTAGTTGCCATCATTAAGTTGGGCACTCTAAGTTGACTGCCGGTGACAAACCGGAGGAAGGTGGGGATGACGTCAAATCATCATGCCCCTTATGATTTGGGCTACACACGTGCTACAATGGACAATACAAAGGGCAGCGAAACCGCGAGGTCAAGCAAATCCCATAAAGTTGTTCTCAGTTCGGATTGTAGTCTGCAACTCGACTACATGAAGCTGGAATCGCTAGTAATCGTAGATCAGAATGCTACGGTGAATACGTTCCCGGGTCTTGTACACACCGCCGTCACACCACGAGAGTTTGTAACACCCGAAGCCGGTGGAGTAACC"

def predict_bacterial_species(dna_sequence: str):
    clean_seq = "".join([c for c in dna_sequence.upper() if c in "ACGT"])
    if len(clean_seq) < 50:
        return {"Error": 1.0}, f"❌ Input DNA sequence is too short ({len(clean_seq)} bp). Minimum 50 bp required."

    inputs = tokenizer(clean_seq)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()

    id2label = model.config.id2label
    confidences = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    
    top_species = id2label[pred_id]
    top_score = probs[pred_id].item() * 100
    
    summary_text = (
        f"### 🔬 Genomic Analysis Summary\n"
        f"- **Input DNA Length**: `{len(clean_seq)} bp`\n"
        f"- **Predicted Species**: **{top_species}**\n"
        f"- **Confidence Score**: `{top_score:.2f}%`"
    )
    
    return confidences, summary_text

# Build Gradio UI
with gr.Blocks(title="Bacterial 16S Species Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧬 Bacterial 16S rRNA Species Classifier
        Fine-tuned **DNA BERT Transformer** predicting bacterial species from 16S rRNA gene sequences.
        """
    )
    
    with gr.Row():
        with gr.Column():
            dna_input = gr.Textbox(
                label="Input DNA Sequence (16S rRNA / FASTA)",
                placeholder="Paste nucleotide sequence (A, C, G, T)...",
                lines=10,
                value=SAMPLE_ECOLI
            )
            with gr.Row():
                btn_ecoli = gr.Button("Paste E. coli Sample", size="sm")
                btn_saureus = gr.Button("Paste S. aureus Sample", size="sm")
                btn_clear = gr.Button("Clear", size="sm")
            
            btn_predict = gr.Button("🔬 Classify Bacterial Species", variant="primary")
            
        with gr.Column():
            output_summary = gr.Markdown("### 🔬 Genomic Analysis Summary\nSubmit a sequence to view predictions.")
            output_chart = gr.Label(label="Species Probability Distribution", num_top_classes=5)

    btn_ecoli.click(fn=lambda: SAMPLE_ECOLI, outputs=dna_input)
    btn_saureus.click(fn=lambda: SAMPLE_SAUREUS, outputs=dna_input)
    btn_clear.click(fn=lambda: "", outputs=dna_input)
    
    btn_predict.click(
        fn=predict_bacterial_species,
        inputs=dna_input,
        outputs=[output_chart, output_summary]
    )

if __name__ == "__main__":
    demo.launch()
