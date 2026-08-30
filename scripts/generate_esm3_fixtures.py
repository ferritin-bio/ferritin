#!/usr/bin/env python3
"""Generate ESM3 sm-open parity fixtures for the Rust numerical comparison test.

Runs the EvolutionaryScale reference ESM3 sm-open-v1 on a fixed sequence and
saves the per-residue embeddings as safetensors. The Rust test
`test_esm3_parity_vs_python_reference` (marked #[ignore]) loads this fixture and
asserts per-residue cosine similarity above a floor.

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

# Must match SHORT_SEQ in tests/test_plm_esm3.rs
SEQUENCE = "ACDEFGHIK"


@torch.no_grad()
def get_embeddings(sequence: str) -> torch.Tensor:
    """Return per-residue embeddings, shape (L + 2, 1536), including BOS/EOS."""
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, SamplingConfig

    client = ESM3.from_pretrained("esm3_sm_open_v1")
    client.eval()

    protein = ESMProtein(sequence=sequence)
    tensor = client.encode(protein)
    output = client.forward_and_sample(
        tensor,
        SamplingConfig(return_per_residue_embeddings=True),
    )
    # per_residue_embedding: (L + 2, 1536)
    return output.per_residue_embedding.float().cpu()


def main():
    parser = argparse.ArgumentParser(description="Generate ESM3 sm-open parity fixtures")
    parser.add_argument("--output", required=True, help="Output directory for safetensors")
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding reference sequence ({len(SEQUENCE)} residues) with ESM3 sm-open ...")
    embeddings = get_embeddings(SEQUENCE)
    print(f"  embeddings shape: {tuple(embeddings.shape)} (expected ({len(SEQUENCE) + 2}, 1536))")

    out_path = output_dir / "esm3_parity.safetensors"
    save_file({"embeddings": embeddings.contiguous()}, str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
