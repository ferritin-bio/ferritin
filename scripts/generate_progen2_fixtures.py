#!/usr/bin/env python3
"""Generate the optional ProGen2-small likelihood parity fixture.

The Rust runner scores the residue targets after the explicit ProGen terminal
tokens ``1`` and ``2``. This script mirrors that shift and stores one scalar
log-likelihood per reference sequence as float32 safetensors.

Usage::

    pip install transformers safetensors torch huggingface_hub tokenizers
    python scripts/generate_progen2_fixtures.py \
        --output crates/ferritin-plms/tests/fixtures/
"""

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

MODEL_ID = "hugohrban/progen2-small"
SEQUENCES = {
    "ubiquitin_nterm": "MQIFVKTLTGK",
    "glycine_repeat": "GGGGGGGGG",
    "alt_charged": "KEKEKEKEK",
}


def score(model, tokenizer: Tokenizer, sequence: str) -> torch.Tensor:
    ids = tokenizer.encode(f"1{sequence}2").ids
    assert ids[0] == 3 and ids[-1] == 4
    assert all(5 <= token <= 29 for token in ids[1:-1])
    input_ids = torch.tensor([ids], dtype=torch.long)
    logits = model(input_ids=input_ids).logits[0].float()
    residue_logits = logits[:-2, 5:30]
    residue_targets = torch.tensor(ids[1:-1], dtype=torch.long) - 5
    return torch.log_softmax(residue_logits, dim=-1).gather(
        1, residue_targets.unsqueeze(1)
    ).sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = hf_hub_download(args.model, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    with torch.no_grad():
        tensors = {
            f"{name}_log_likelihood": score(model, tokenizer, sequence)
            for name, sequence in SEQUENCES.items()
        }
    path = output / "progen2_parity.safetensors"
    save_file(tensors, str(path))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
