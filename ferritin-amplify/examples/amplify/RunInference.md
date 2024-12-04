# Run Inference

From the HF Repo: https://huggingface.co/chandar-lab/AMPLIFY_120M


```shell
python3 -m venv env
source env/bin/activate
pip install transformers datasets xformers biopython
```

```shell
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz
```


```python
from transformers import AutoModel
from transformers import AutoTokenizer
from datasets import load_dataset

# Load AMPLIFY and tokenizer
model = AutoModel.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True)

# Move the model to GPU (required due to Flash Attention)
model = model.to("cuda")

# Load the UniProt validation set
dataset = load_dataset("chandar-lab/UR100P", data_dir="UniProt", split="test", cache_dir="/workspace/datasets")

for sample in dataset:
    # Protein
    print("Sample: ", sample["name"], sample["sequence"])

    # Tokenize the protein
    input = tokenizer.encode(sample["sequence"], return_tensors="pt")
    print("Input: ", input)
    
    # Move to the GPU and make a prediction
    input = input.to("cuda")
    output = model(input)
    print("Output: ", output)
    
    break

```


```python
import torch
import torch.nn.functional as F


input = tokenizer.encode("METVAL", return_tensors="pt")
input = input.to("cuda")
output = model(input) 

# get top value
max_indices = torch.argmax(output.logits[0], dim=-1)
decoded = tokenizer.batch_decode(max_indices)

# get topk values
softmax_probs = F.softmax(output.logits[0], dim=-1)
k = 3
top_k_probs, top_k_indices = torch.topk(softmax_probs, k=k, dim=-1)

for (k, i) in zip(top_k_probs, top_k_indices):
    print(f"Decoder: {tokenizer.decode(i)}")
```


```python
from Bio import SeqIO

rec = [s for i, s in enumerate(SeqIO.parse("uniprot_sprot.fasta", "fasta")) if i == 0][0]
print(rec.seq)
input = tokenizer.encode(str(rec.seq), return_tensors="pt")
input = input.to("cuda")
output = model(input) 
max_indices = torch.argmax(output.logits[0], dim=-1)
decoded = tokenizer.batch_decode(max_indices)

print(rec.seq)
print("".join(decoded))

# inputs --> outputs
```