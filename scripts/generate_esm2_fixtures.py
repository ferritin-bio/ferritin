#!/usr/bin/env python3
"""Generate ESM-2 parity fixtures for Rust numerical comparison tests.

Runs HuggingFace EsmForMaskedLM (esm2_t6_8M_UR50D) on a fixed reference set
and saves per-position logits as safetensors.  The Rust test (marked #[ignore])
loads these fixtures and asserts logit agreement within tolerance 1e-3.

Usage
-----
    pip install transformers safetensors torch

    python scripts/generate_esm2_fixtures.py \
        --output crates/ferritin-test-data/data/safetensors/esm2_parity/

Reference sequences (from ferritin-lgr issue):
    - Ubiquitin N-term: MQIFVKTLTGK
    - Glycine repeat:   GGGGGGGGG   (9 Gly)
    - Alt-charged:      KEKEKEKEK   (9 residues)
    - Masked token:     MQ[MASK]FVKTLTGK

The model used is esm2_t6_8M (smallest ESM-2 variant, T6_8M in Rust).
"""

import argparse
from pathlib import Path

import torch
from transformers import EsmForMaskedLM, EsmTokenizer

MODEL_ID = "facebook/esm2_t6_8M_UR50D"

SEQUENCES = {
    "ubiquitin_nterm": "MQIFVKTLTGK",
    "glycine_repeat": "GGGGGGGGG",
    "alt_charged": "KEKEKEKEK",
    # Masked sequence — replace one residue with the mask token
    "masked_seq": "MQ<mask>FVKTLTGK",
}


@torch.no_grad()
def get_logits(model, tokenizer, sequence: str) -> torch.Tensor:
    """Return per-position logits, shape (L, vocab_size), excluding BOS/EOS."""
    inputs = tokenizer(sequence, return_tensors="pt")
    outputs = model(**inputs)
    # Strip the BOS and EOS positions
    logits = outputs.logits[0, 1:-1, :]  # (L, vocab_size)
    return logits.float()


def main():
    parser = argparse.ArgumentParser(description="Generate ESM-2 parity fixtures")
    parser.add_argument("--model", default=MODEL_ID, help="HuggingFace ESM2 model ID")
    parser.add_argument("--output", required=True, help="Output directory for safetensors")
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    tokenizer = EsmTokenizer.from_pretrained(args.model)
    model = EsmForMaskedLM.from_pretrained(args.model)
    model.eval()

    tensors = {}
    for name, seq in SEQUENCES.items():
        print(f"  Scoring {name!r}: {seq}")
        logits = get_logits(model, tokenizer, seq)
        tensors[f"{name}_logits"] = logits
        print(f"    logits shape: {logits.shape}")

    out_path = output_dir / "esm2_parity.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
