"""
Download Real Bacterial Genomic Sequences from NCBI Entrez API / Curated Benchmark
Downloads 16S rRNA / genomic DNA sequences for multiple bacterial species.
"""

import os
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

SPECIES_LIST = [
    {"name": "Escherichia coli", "label": 0, "term": "Escherichia coli[Organism] AND 16S ribosomal RNA[Title]" },
    {"name": "Bacillus subtilis", "label": 1, "term": "Bacillus subtilis[Organism] AND 16S ribosomal RNA[Title]" },
    {"name": "Staphylococcus aureus", "label": 2, "term": "Staphylococcus aureus[Organism] AND 16S ribosomal RNA[Title]" },
    {"name": "Pseudomonas aeruginosa", "label": 3, "term": "Pseudomonas aeruginosa[Organism] AND 16S ribosomal RNA[Title]" },
    {"name": "Salmonella enterica", "label": 4, "term": "Salmonella enterica[Organism] AND 16S ribosomal RNA[Title]" }
]

DATA_DIR = Path(__file__).parent / "data"

def fetch_ncbi_sequences(term, max_results=25):
    """Fetch DNA FASTA sequences from NCBI Nucleotide DB with rate-limit handling."""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode({
        "db": "nucleotide",
        "term": term,
        "retmode": "json",
        "retmax": max_results
    })
    
    req = urllib.request.Request(search_url, headers={"User-Agent": "BacterialGenomicsApp/1.0 (contact@example.com)"})
    time.sleep(1.0) # NCBI rate limit politeness
    
    try:
        with urllib.request.urlopen(req) as resp:
            search_data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"    NCBI Search error: {e}")
        return []
    
    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []
    
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode({
        "db": "nucleotide",
        "id": ",".join(id_list),
        "rettype": "fasta",
        "retmode": "text"
    })
    
    req = urllib.request.Request(fetch_url, headers={"User-Agent": "BacterialGenomicsApp/1.0 (contact@example.com)"})
    time.sleep(1.0)
    
    try:
        with urllib.request.urlopen(req) as resp:
            fasta_text = resp.read().decode()
    except Exception as e:
        print(f"    NCBI Fetch error: {e}")
        return []
    
    sequences = []
    current_seq = []
    for line in fasta_text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_seq:
                full_seq = "".join(current_seq).upper()
                clean_seq = "".join([c for c in full_seq if c in "ACGT"])
                if len(clean_seq) >= 200:
                    sequences.append(clean_seq[:1000])
                current_seq = []
        else:
            current_seq.append(line)
            
    if current_seq:
        full_seq = "".join(current_seq).upper()
        clean_seq = "".join([c for c in full_seq if c in "ACGT"])
        if len(clean_seq) >= 200:
            sequences.append(clean_seq[:1000])
            
    return sequences

def main():
    DATA_DIR.mkdir(exist_ok=True)
    all_dataset = []
    
    print("Fetching real bacterial DNA sequences from NCBI...")
    for species in SPECIES_LIST:
        print(f"  Downloading sequences for {species['name']}...")
        seqs = fetch_ncbi_sequences(species["term"], max_results=100)
        print(f"    Fetched {len(seqs)} sequences for {species['name']}.")
        for seq in seqs:
            all_dataset.append({
                "sequence": seq,
                "label": species["label"],
                "species": species["name"]
            })
            
    output_file = DATA_DIR / "bacterial_16S_sequences.json"
    with open(output_file, "w") as f:
        json.dump(all_dataset, f, indent=2)
        
    print(f"\nDataset prepared with {len(all_dataset)} real bacterial sequences across {len(SPECIES_LIST)} species.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
