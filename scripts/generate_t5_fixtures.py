#!/usr/bin/env python3
"""Generate T5-family parity fixtures (ProtT5 and Ankh) for the Rust tests.

Runs `Rostlab/prot_t5_xl_half_uniref50-enc` through HuggingFace `T5EncoderModel`
and saves the per-residue encoder hidden states as safetensors. The Rust test
(`tests/test_plm_prott5.rs`, `#[ignore]`d) loads these and asserts agreement.

Two things this script pins down that are easy to get wrong (ferritin-goh.5):

* **Whitespace.** ProtT5's SentencePiece pieces are all `<boundary><letter>`,
  so the reference pipeline spaces residues out. The fixture is generated from
  the spaced form, which is what Rostlab's model card does; the Rust tokenizer
  skips whitespace and so accepts either form. If the two disagreed, every
  residue would shift by one.

ProtT5's tokenization goes through `sentencepiece` directly rather than through
`transformers.T5Tokenizer`, because loading these SentencePiece files through
transformers is version- and dependency-sensitive:

* transformers **5.x** cannot load them at all — its slow-to-fast converter
  misidentifies `spiece.model` as a tiktoken vocabulary and dies in
  `load_tiktoken_bpe`.
* transformers **4.x** loads them, but only with `protobuf` installed, and
  fails with a bare ImportError otherwise.

Reading `spiece.model` with the library that wrote it sidesteps both. ProstT5
does use `T5Tokenizer` (it needs `added_tokens.json` for the 3Di states, which
live outside `spiece.model`), so generating that fixture needs transformers 4.x
and protobuf; the assertion on token count will catch it if the tokenizer ever
returns something unexpected.
* **Special tokens.** T5 appends `</s>` and prepends nothing. The saved
  tensors have that row stripped, so row `i` is residue `i` — matching
  `PlmRunner::embed_residues` rather than `embed`.

The stored tensors are float32 even though the checkpoint is float16: the
fixture is the reference, and rounding it to F16 would bake the very error the
comparison is trying to measure.

Regenerating is not free of churn: the same script under a different torch
build moves the embeddings by ~1e-6 (ids are unaffected). That is float noise,
not a change in the reference, so prefer leaving a committed fixture alone
rather than regenerating it incidentally while adding a new one.

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
ANKH_ID = "ElnaggarLab/ankh-base"
PROSTT5_ID = "Rostlab/ProstT5_fp16"

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


@torch.no_grad()
def embed_ankh(model, tok, sequence: str) -> torch.Tensor:
    """Per-residue Ankh encoder states, shape (L, d_model), EOS stripped.

    Ankh ships a real `tokenizer.json` whose Unigram vocabulary has NO
    SentencePiece boundary marker, so unlike ProtT5 it takes the bare sequence
    with no spacing. Its ids are nonetheless the same alphabet at the same
    positions as ProtT5's, which is why one Rust table serves both — asserted
    here rather than assumed.
    """
    enc = tok(sequence, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"]
    assert ids.shape[1] == len(sequence) + 1, (
        f"expected {len(sequence)} residues + 1 EOS, got {ids.shape[1]} tokens"
    )
    assert ids[0, -1].item() == tok.eos_token_id, "last token should be </s>"
    out = model(input_ids=ids, attention_mask=enc["attention_mask"])
    return out.last_hidden_state[0, : len(sequence), :].float(), ids[0].to(torch.int32)


def generate_ankh(output_dir, model_id: str):
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, T5EncoderModel

    print(f"Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = T5EncoderModel.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    tensors = {}
    for name, seq in SEQUENCES.items():
        emb, ids = embed_ankh(model, tok, seq)
        tensors[f"{name}_embeddings"] = emb
        tensors[f"{name}_input_ids"] = ids
        print(f"  {name!r}: embeddings {tuple(emb.shape)}  ids {ids.tolist()}")

    out_path = output_dir / "ankh_parity.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {out_path}")


@torch.no_grad()
def generate_prostt5(output_dir, model_id: str):
    """Reference AA->3Di translations, plus the exact input ids they came from.

    ProstT5 shares ProtT5's SentencePiece container, so tokenization goes
    through `sentencepiece` directly for the same reason (see the module
    docstring) — but the 3Di states and the two direction tokens live in
    `added_tokens.json`, outside spiece.model, so they are added here.

    Greedy decoding with no sampling, matching Rostlab's own usage: the mapping
    is close to deterministic and sampling would only add noise to a structural
    annotation.
    """
    import json

    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    # ProstT5's own tokenizer, not a reconstruction of it. This needs
    # `protobuf` installed (transformers converts spiece.model through it) and
    # transformers 4.x — the 5.x converter misidentifies these SentencePiece
    # files as tiktoken and dies in load_tiktoken_bpe.
    tok = T5Tokenizer.from_pretrained(model_id, legacy=False)
    with open(hf_hub_download(model_id, "added_tokens.json")) as fh:
        added = json.load(fh)
    by_id = {v: k for k, v in added.items()}

    model = T5ForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    tensors = {}
    for name, seq in SEQUENCES.items():
        if set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
            # ProstT5 translates the 20 standard residues; the rare-residue
            # probe is a ProtT5/Ankh tokenizer case, not a structural one.
            continue
        # The direction prefix, then the spaced residues — Rostlab's own
        # recipe. Spacing matters: every SentencePiece piece carries a word
        # boundary, so an unspaced sequence reads as one unknown word.
        prompt = "<AA2fold> " + " ".join(list(seq))
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        assert len(ids) == len(seq) + 2, (
            f"{name}: expected prefix + {len(seq)} residues + EOS, got {ids}"
        )
        input_ids = torch.tensor([ids], dtype=torch.long)
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=len(seq),
            min_new_tokens=len(seq),
            do_sample=False,
            num_beams=1,
        )
        # generate() re-emits decoder_start_token_id first; drop it.
        gen = [int(t) for t in out[0].tolist()[1:]][: len(seq)]
        # 3Di states live in added_tokens.json, outside spiece.model, so decode
        # them through that map.
        text = "".join(by_id.get(t, "?") for t in gen)
        assert len(text) == len(seq), f"{name}: {len(text)} states for {len(seq)} residues"
        assert "?" not in text, f"{name}: non-3Di token in {gen}"
        tensors[f"{name}_input_ids"] = torch.tensor(ids, dtype=torch.int32)
        tensors[f"{name}_3di_ids"] = torch.tensor(gen, dtype=torch.int32)
        print(f"  {name!r}: {seq} -> {text}")

    out_path = output_dir / "prostt5_parity.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {out_path}")


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

    generate_ankh(output_dir, ANKH_ID)
    print(f"Loading {PROSTT5_ID} ...")
    generate_prostt5(output_dir, PROSTT5_ID)


if __name__ == "__main__":
    main()
