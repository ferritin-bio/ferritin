#!/usr/bin/env python3
"""Generate ProtT5 parity fixtures for the Rust numerical comparison test.

Runs `Rostlab/prot_t5_xl_half_uniref50-enc` through HuggingFace `T5EncoderModel`
and saves the per-residue encoder hidden states as safetensors. The Rust test
(`tests/test_plm_prott5.rs`, `#[ignore]`d) loads these and asserts agreement.

Two things this script pins down that are easy to get wrong (ferritin-goh.5):

* **Whitespace.** ProtT5's SentencePiece pieces are all `<boundary><letter>`,
  so the reference pipeline spaces residues out. The fixture is generated from
  the spaced form, which is what Rostlab's model card does; the Rust tokenizer
  skips whitespace and so accepts either form. If the two disagreed, every
  residue would shift by one.

Tokenization goes through `sentencepiece` directly rather than through
`transformers.T5Tokenizer`. transformers 5.x cannot load this repo's
`spiece.model` at all — its slow-to-fast converter misidentifies the file as a
tiktoken vocabulary and dies in `load_tiktoken_bpe`. Reading `spiece.model` with
the library that wrote it is both a working path and a stronger reference: it
depends on the checkpoint rather than on a transformers version.
* **Special tokens.** T5 appends `</s>` and prepends nothing. The saved
  tensors have that row stripped, so row `i` is residue `i` — matching
  `PlmRunner::embed_residues` rather than `embed`.

The stored tensors are float32 even though the checkpoint is float16: the
fixture is the reference, and rounding it to F16 would bake the very error the
comparison is trying to measure.

Usage
-----
    pip install transformers safetensors torch sentencepiece huggingface_hub

    python scripts/generate_prott5_fixtures.py \
        --output crates/ferritin-plms/tests/fixtures/
"""

import argparse
from pathlib import Path

import torch
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from transformers import T5EncoderModel

MODEL_ID = "Rostlab/prot_t5_xl_half_uniref50-enc"

# Short on purpose: the fixture is committed, and d_model is 1024, so a 76-mer
# would be a 300 KB tensor for no extra signal.
SEQUENCES = {
    "ubiquitin_nterm": "MQIFVKTLTGK",
    "glycine_repeat": "GGGGGGG",
    "alt_charged": "KEKEKEK",
    # Exercises the four residues Rostlab's model card rewrites to X. They have
    # their own embedding rows and this port does NOT rewrite them, so a fixture
    # that silently did would disagree.
    "rare_residues": "MUZOBX",
}


@torch.no_grad()
def encode(sp, sequence: str) -> list:
    """Token ids for `sequence`: SentencePiece over the spaced form, then EOS.

    Asserts the two properties the Rust side declares, rather than trusting
    them: exactly one piece per residue (so the spacing convention is doing its
    job), and every piece carrying SentencePiece's word-boundary marker.
    """
    spaced = " ".join(list(sequence))
    ids = sp.encode(spaced)
    assert len(ids) == len(sequence), (
        f"{sequence!r} produced {len(ids)} pieces for {len(sequence)} residues; "
        "the whitespace convention is wrong"
    )
    for residue, i in zip(sequence, ids):
        piece = sp.id_to_piece(i)
        assert piece == "\u2581" + residue, f"piece {piece!r} for residue {residue!r}"
    return ids + [sp.eos_id()]


@torch.no_grad()
def embed(model, sp, sequence: str) -> torch.Tensor:
    """Per-residue encoder hidden states, shape (L, d_model), EOS stripped."""
    ids = encode(sp, sequence)
    # T5: no BOS, one trailing </s>. This is the EOS_ONLY layout the Rust
    # SpecialTokenLayout declares.
    assert len(ids) == len(sequence) + 1
    assert ids[-1] == sp.eos_id()
    input_ids = torch.tensor([ids], dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    return out.last_hidden_state[0, : len(sequence), :].float()


def main():
    parser = argparse.ArgumentParser(description="Generate ProtT5 parity fixtures")
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
    sp = spm.SentencePieceProcessor(
        model_file=hf_hub_download(args.model, "spiece.model")
    )
    # float32 on CPU: the reference should not inherit the checkpoint's F16
    # rounding, which is what the Rust side is being measured against.
    model = T5EncoderModel.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    tensors = {}
    for name, seq in SEQUENCES.items():
        print(f"  Embedding {name!r}: {seq}")
        tensors[f"{name}_embeddings"] = embed(model, sp, seq)
        # Committed alongside the embeddings so a tokenizer regression is
        # diagnosable without re-running Python.
        tensors[f"{name}_input_ids"] = torch.tensor(encode(sp, seq), dtype=torch.int32)
        print(f"    embeddings {tuple(tensors[f'{name}_embeddings'].shape)}"
              f"  ids {tensors[f'{name}_input_ids'].tolist()}")

    out_path = output_dir / "prott5_parity.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
