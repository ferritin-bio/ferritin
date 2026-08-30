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
from transformers import AutoModel, AutoTokenizer

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
    # AMPLIFY expects an *additive* attention mask (0.0 keep / -inf drop), not
    # the tokenizer's 0/1 mask. For a single unpadded sequence every token is
    # kept, so this is an all-zeros (no-op) bias.
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.where(
            inputs["attention_mask"].bool(), 0.0, float("-inf")
        )
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
    # AMPLIFY ships its modeling code in the HF repo, so recent transformers
    # require trust_remote_code=True to load it.
    # AMPLIFY registers its class as AutoModel (config.json auto_map), and that
    # class already returns a MaskedLMOutput with per-position logits.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # low_cpu_mem_usage=False forces eager init so AMPLIFY's non-persistent
    # `freqs_cis` RoPE buffer is materialised on CPU (not left on the meta
    # device, which recent transformers uses by default).
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, low_cpu_mem_usage=False
    )
    model.eval()

    # `freqs_cis` is a plain (non-buffer) attribute built in __init__, so under
    # transformers' meta-init it can be left on the meta device. Rebuild it on
    # CPU from the model's own precompute_freqs_cis so RoPE has real data.
    if getattr(model, "freqs_cis", None) is not None and model.freqs_cis.is_meta:
        import importlib

        pkg = type(model).__module__.rsplit(".", 1)[0]
        precompute_freqs_cis = importlib.import_module(f"{pkg}.rotary").precompute_freqs_cis
        model.freqs_cis = precompute_freqs_cis(
            model.config.hidden_size // model.config.num_attention_heads,
            model.config.max_length,
        )

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
