#!/usr/bin/env python3
"""Generate ESM-2-family parity fixtures for Rust numerical comparison tests.

Runs a HuggingFace EsmForMaskedLM on a fixed reference set and saves
per-position logits as safetensors. The Rust tests (marked #[ignore]) load
these fixtures and assert logit agreement within tolerance.

Two variants, because "ESM-2 family" is a claim about the backbone, not about
the alphabet:

``--variant esm2`` (default)
    facebook/esm2_t6_8M_UR50D over the stock 33-token alphabet, one character
    per residue. Produces ``esm2_parity.safetensors``.

``--variant saprot``
    westlake-repl/SaProt_35M_AF2 over the 446-token (amino acid, 3Di) product
    alphabet, **two** characters per residue. Produces
    ``saprot_parity.safetensors``. Same EsmForMaskedLM backbone, completely
    different tokenizer path — which is exactly why it needs its own fixture
    rather than inheriting trust from the esm2 row (ferritin-100.34).

Usage
-----
    pip install transformers safetensors torch

    python scripts/generate_esm2_fixtures.py \
        --output crates/ferritin-plms/tests/fixtures/

    python scripts/generate_esm2_fixtures.py --variant saprot \
        --output crates/ferritin-plms/tests/fixtures/

Reference sequences (from ferritin-lgr issue):
    - Ubiquitin N-term: MQIFVKTLTGK
    - Glycine repeat:   GGGGGGGGG   (9 Gly)
    - Alt-charged:      KEKEKEKEK   (9 residues)
    - Masked token:     MQ[MASK]FVKTLTGK

The default model is esm2_t6_8M (smallest ESM-2 variant, T6_8M in Rust).
"""

import argparse
from pathlib import Path

import torch
from transformers import EsmForMaskedLM, EsmTokenizer

MODEL_ID = "facebook/esm2_t6_8M_UR50D"
SAPROT_MODEL_ID = "westlake-repl/SaProt_35M_AF2"

SEQUENCES = {
    "ubiquitin_nterm": "MQIFVKTLTGK",
    "glycine_repeat": "GGGGGGGGG",
    "alt_charged": "KEKEKEKEK",
    # Masked sequence — replace one residue with the mask token
    "masked_seq": "MQ<mask>FVKTLTGK",
}


def interleave(aa: str, threedi: str) -> str:
    """Weave an amino-acid string and a 3Di string into SaProt's alphabet.

    SaProt reads ``(residue, state)`` pairs, so "MQ" + "dv" is "MdQv" and not
    "MQdv". Mirrors `interleave` in src/esm2/saprot_tokenizer.rs.
    """
    if len(aa) != len(threedi):
        raise ValueError(f"length mismatch: {len(aa)} residues vs {len(threedi)} 3Di states")
    return "".join(a + t for a, t in zip(aa, threedi))


# The 3Di strings here are ProstT5's real output for the paired sequence, taken
# from tests/test_saprot_bridge.rs, so these are pairs SaProt's vocabulary
# actually contains rather than plausible-looking ones. A pair that merely
# looks right encodes to <unk> and the parity test would then be comparing
# two models' opinions about unknown tokens.
SAPROT_SEQUENCES = {
    "ubiquitin_nterm": interleave("MQIFVKTLTGK", "dvvvvcvvvvd"),
    "glycine_repeat": interleave("GGGGGGGGG", "ddddddddd"),
    "alt_charged": interleave("KEKEKEKEK", "vvvvvvvvv"),
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
    parser = argparse.ArgumentParser(description="Generate ESM-2-family parity fixtures")
    parser.add_argument(
        "--variant",
        default="esm2",
        choices=["esm2", "saprot"],
        help="which alphabet to score over (default: esm2)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID (default: the variant's own model)",
    )
    parser.add_argument("--output", required=True, help="Output directory for safetensors")
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    if args.variant == "saprot":
        model_id = args.model or SAPROT_MODEL_ID
        sequences = SAPROT_SEQUENCES
        fixture_name = "saprot_parity.safetensors"
        chars_per_residue = 2
    else:
        model_id = args.model or MODEL_ID
        sequences = SEQUENCES
        fixture_name = "esm2_parity.safetensors"
        chars_per_residue = 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_id} ...")
    tokenizer = EsmTokenizer.from_pretrained(model_id)
    model = EsmForMaskedLM.from_pretrained(model_id)
    model.eval()

    tensors = {}
    for name, seq in sequences.items():
        print(f"  Scoring {name!r}: {seq}")
        logits = get_logits(model, tokenizer, seq)

        # Guard the alphabet, not just the shape. If SaProt's vocab.txt were
        # read one character per residue the sequence would still tokenize --
        # into twice as many rows, every other one <unk> -- and the fixture
        # would look perfectly well-formed. Assert the row count matches the
        # residue count so that failure is loud here rather than becoming a
        # confusing mismatch on the Rust side (ferritin-100.34).
        expected_rows = len(seq) // chars_per_residue
        if logits.shape[0] != expected_rows:
            raise SystemExit(
                f"{name}: got {logits.shape[0]} logit rows for {len(seq)} characters, "
                f"expected {expected_rows} at {chars_per_residue} char(s) per residue. "
                f"The tokenizer is not reading this alphabet the way {args.variant} requires."
            )

        unk = tokenizer.unk_token_id
        ids = tokenizer(seq, return_tensors="pt")["input_ids"][0].tolist()
        if unk is not None and unk in ids:
            raise SystemExit(
                f"{name}: sequence contains token(s) absent from {model_id}'s vocabulary "
                f"(<unk> at {[i for i, t in enumerate(ids) if t == unk]}). A parity fixture "
                f"over <unk> rows compares two models' opinions about nothing."
            )

        tensors[f"{name}_logits"] = logits
        print(f"    logits shape: {tuple(logits.shape)} ({expected_rows} residues)")

    out_path = output_dir / fixture_name
    save_file(tensors, str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
