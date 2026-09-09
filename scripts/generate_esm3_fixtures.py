#!/usr/bin/env python3
"""Generate ESM3 parity fixtures for the Rust numerical comparison tests.

Emits two fixtures:

``esm3_parity.safetensors``
    Per-residue embeddings from ESM3 sm-open-v1 on a fixed sequence. The Rust
    test `test_esm3_parity_vs_python_reference` asserts cosine similarity above
    a floor.

``esm3_structure_parity.safetensors``
    Structure token ids from the VQ-VAE `StructureTokenEncoder` on a real
    backbone, plus the backbone coordinates they came from (ferritin-100.27).
    Tokens are discrete codebook indices, so the Rust test asserts **exact
    equality** — a tolerance would be meaningless.

    The coordinates are saved alongside the tokens deliberately. If the Rust
    test rebuilt the backbone itself, the two sides could disagree about the
    input rather than the model, and a float difference in parsing would look
    like a parity failure. Loading the same array both sides removes that.

Gated access
------------
esm3-sm-open-v1 is gated: accept the Cambrian Non-Commercial license at
<https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1> and run
``huggingface-cli login`` before running this script.

Alignment contract (must match tests/support/parity.rs)
-------------------------------------------------------
The saved ``embeddings`` tensor has shape ``(L + 2, 1536)`` and **keeps** the
BOS/EOS rows. The Rust side compares with ``SpecialTokens::NONE`` (batch dim
squeezed, no rows stripped), so a BOS/EOS placement mismatch surfaces as a
failure at position 0 or L+1 rather than passing silently.

Usage
-----
    pip install esm safetensors torch

    python scripts/generate_esm3_fixtures.py \
        --output crates/ferritin-plms/tests/fixtures/

The reference sequence must stay in sync with ``SHORT_SEQ`` in
crates/ferritin-plms/tests/test_plm_esm3.rs.
"""

import argparse
from pathlib import Path

import torch

# A real backbone rather than an idealised helix. The existing Rust test builds
# an ideal alpha-helix, which is locally periodic and therefore exercises only a
# narrow slice of the codebook; the defects this fixture exists to catch — a
# transposed rotation in the gathered frames, an off-by-one in the
# relative-position bin shift, the wrong neighbour taken as the query node —
# are all geometry-dependent and can hide in a degenerate input.
# 1BC8 is a protein/DNA complex: chains A and B are DNA (residues `DT`, `DA`,
# … with no CA at all) and chain C is the 93-residue protein. Picking chain A
# here would silently yield an empty backbone.
BACKBONE_PDB = "crates/ferritin-test-data/data/structures/1BC8.pdb"
BACKBONE_CHAIN = "C"

# Must match SHORT_SEQ in tests/test_plm_esm3.rs
SEQUENCE = "ACDEFGHIK"


@torch.no_grad()
def get_embeddings(sequence: str) -> torch.Tensor:
    """Return per-residue embeddings, shape (L + 2, 1536), including BOS/EOS."""
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, SamplingConfig

    # Upcast to float32 before running (ferritin-100.29).
    #
    # The checkpoint is BFloat16, but ESM3.forward synthesises float32
    # auxiliary tensors when the caller supplies no structure — the plddt
    # values feeding `EncodeInputs.plddt_projection`, and the identity frames
    # feeding geometric attention. In esm 3.4.0 / torch 2.11 each meets a
    # BFloat16 weight and raises "expected m1 and m2 to have the same dtype".
    # Both are upstream SDK bugs, not ferritin's.
    #
    # Upcasting fixes the class rather than the two instances, and it is also
    # the reference we actually want: the Rust side loads these same weights at
    # F32, so a BFloat16 reference would fold BFloat16 rounding into the thing
    # parity is measured against. bf16 -> f32 is lossless, so no checkpoint
    # information is invented by doing this.
    client = ESM3.from_pretrained("esm3_sm_open_v1")
    client.eval()
    client = client.to(torch.float32)

    protein = ESMProtein(sequence=sequence)
    tensor = client.encode(protein)
    output = client.forward_and_sample(
        tensor,
        SamplingConfig(return_per_residue_embeddings=True),
    )
    # per_residue_embedding: (L + 2, 1536)
    return output.per_residue_embedding.float().cpu()


def read_backbone(path: str, chain: str) -> torch.Tensor:
    """Backbone N/CA/C coordinates from a PDB file, shape (L, 3, 3).

    A deliberately minimal parser: the fixture must depend on the checkpoint,
    not on whichever structure library happens to be installed. Only the first
    altloc of each atom is taken, and a residue is emitted only when all three
    backbone atoms are present — a residue missing its N would otherwise become
    silent garbage geometry.
    """
    residues: dict = {}
    order: list = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            atom = line[12:16].strip()
            if atom not in ("N", "CA", "C"):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            key = line[22:27]  # residue sequence number + insertion code
            if key not in residues:
                residues[key] = {}
                order.append(key)
            residues[key].setdefault(
                atom,
                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            )

    coords = [
        [residues[k]["N"], residues[k]["CA"], residues[k]["C"]]
        for k in order
        if {"N", "CA", "C"} <= residues[k].keys()
    ]
    dropped = len(order) - len(coords)
    if dropped:
        print(f"  dropped {dropped} residue(s) missing a backbone atom")
    if not coords:
        raise SystemExit(
            f"no N/CA/C backbone found in {path} chain {chain}. "
            "Nucleic-acid chains have no CA — check the chain id."
        )
    return torch.tensor(coords, dtype=torch.float32)


@torch.no_grad()
def get_structure_tokens(coords: torch.Tensor) -> torch.Tensor:
    """Structure token ids for a backbone, shape (L,).

    `StructureTokenEncoder.encode` returns `(z_q, min_encoding_indices)`; the
    ids are the second. No special tokens are added — the encoder emits exactly
    one token per residue, which is the contract the Rust side asserts.
    """
    from esm.pretrained import ESM3_structure_encoder_v0

    encoder = ESM3_structure_encoder_v0("cpu")
    encoder.eval()
    _, indices = encoder.encode(coords.unsqueeze(0))
    return indices[0].to(torch.int32).cpu()


def main():
    parser = argparse.ArgumentParser(description="Generate ESM3 parity fixtures")
    parser.add_argument("--output", required=True, help="Output directory for safetensors")
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The two fixtures are generated independently: they exercise different
    # checkpoints and different SDK entry points, so a failure in one should
    # not cost the other. Both currently generate.
    failures = []

    try:
        print(f"Embedding reference sequence ({len(SEQUENCE)} residues) with ESM3 sm-open ...")
        embeddings = get_embeddings(SEQUENCE)
        print(f"  embeddings shape: {tuple(embeddings.shape)} "
              f"(expected ({len(SEQUENCE) + 2}, 1536))")
        out_path = output_dir / "esm3_parity.safetensors"
        save_file({"embeddings": embeddings.contiguous()}, str(out_path))
        print(f"Saved {out_path}")
    except Exception as e:  # noqa: BLE001 - report and continue to the next fixture
        failures.append(f"esm3_parity: {type(e).__name__}: {e}")
        print(f"  FAILED: {type(e).__name__}: {e}")

    try:
        print(f"Reading backbone from {BACKBONE_PDB} chain {BACKBONE_CHAIN} ...")
        coords = read_backbone(BACKBONE_PDB, BACKBONE_CHAIN)
        print(f"  backbone shape: {tuple(coords.shape)}")
        tokens = get_structure_tokens(coords)
        print(f"  tokens shape: {tuple(tokens.shape)}, "
              f"range [{int(tokens.min())}, {int(tokens.max())}], "
              f"{len(set(tokens.tolist()))} distinct")
        assert tokens.shape[0] == coords.shape[0], "one structure token per residue"
        assert int(tokens.max()) < 4096, "tokens must index the 4096-entry codebook"
        out_path = output_dir / "esm3_structure_parity.safetensors"
        save_file(
            {
                "backbone_coords": coords.contiguous(),
                "structure_tokens": tokens.contiguous(),
            },
            str(out_path),
        )
        print(f"Saved {out_path}")
    except Exception as e:  # noqa: BLE001 - report alongside any earlier failure
        failures.append(f"esm3_structure_parity: {type(e).__name__}: {e}")
        print(f"  FAILED: {type(e).__name__}: {e}")

    if failures:
        raise SystemExit("\n".join(["Some fixtures were not generated:", *failures]))


if __name__ == "__main__":
    main()
