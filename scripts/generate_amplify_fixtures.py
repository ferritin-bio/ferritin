#!/usr/bin/env python3
"""Generate AMPLIFY parity fixtures for Rust numerical comparison tests.

Runs the chandar-lab AMPLIFY model (AMP120M) via HuggingFace on a fixed set of
reference sequences and saves the per-position logit tensors as safetensors.
The Rust test (marked #[ignore]) loads these fixtures and asserts logit agreement
within tolerance 1e-3.

Usage
-----
    pip install transformers safetensors torch

    python scripts/generate_amplify_fixtures.py \
        --output crates/ferritin-test-data/data/safetensors/amplify_parity/

Reference sequences (from ferritin-1ti issue):
    - Ubiquitin N-term: MQIFVKTLTGK
    - Glycine repeat:   GGGGGGG
    - Alt-charged:      KEKEKEK
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "chandar-lab/AMPLIFY_120M"

SEQUENCES = {
    "ubiquitin_nterm": "MQIFVKTLTGK",
    "glycine_repeat": "GGGGGGG",
    "alt_charged": "KEKEKEK",
}


@torch.no_grad()
def get_logits(model, tokenizer, sequence: str) -> torch.Tensor:
    """Return per-position logits for a sequence, shape (L, vocab_size)."""
    inputs = tokenizer(sequence, return_tensors="pt")
    outputs = model(**inputs)
    # outputs.logits shape: (1, L+2, vocab_size) — strip BOS/EOS tokens
    logits = outputs.logits[0, 1:-1, :]  # (L, vocab_size)
    return logits.float()


def main():
    parser = argparse.ArgumentParser(description="Generate AMPLIFY parity fixtures")
    parser.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--output", required=True, help="Output directory for safetensors")
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.eval()

    tensors = {}
    for name, seq in SEQUENCES.items():
        print(f"  Scoring {name!r}: {seq}")
        logits = get_logits(model, tokenizer, seq)
        tensors[f"{name}_logits"] = logits
        print(f"    logits shape: {logits.shape}")

    out_path = output_dir / "amplify_parity.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
