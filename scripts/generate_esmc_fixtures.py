#!/usr/bin/env python3
"""Generate ESMC-300M parity fixtures for the Rust numerical comparison test.

Runs the EvolutionaryScale reference ESMC-300M on a fixed sequence and saves the
per-residue embeddings as safetensors. The Rust test
`test_esmc_parity_vs_python_reference` (marked #[ignore]) loads this fixture and
asserts per-residue cosine similarity above a floor.

Alignment contract (must match tests/support/parity.rs)
-------------------------------------------------------
The saved ``embeddings`` tensor has shape ``(L + 2, 960)`` and **keeps** the
BOS/EOS rows the tokenizer prepends/appends. The Rust side compares with
``SpecialTokens::NONE`` (batch dim squeezed, no rows stripped), so a mismatch in
BOS/EOS placement surfaces as a failure at position 0 or L+1 rather than passing
silently.

Usage
-----
    pip install esm safetensors torch

    python scripts/generate_esmc_fixtures.py \
        --output crates/ferritin-plms/tests/fixtures/

The reference sequence must stay in sync with ``ESMC_PARITY_SEQ`` in
crates/ferritin-plms/tests/test_plm_esmc.rs.
"""

import argparse
from pathlib import Path

import torch

# Must match ESMC_PARITY_SEQ in tests/test_plm_esmc.rs
SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@torch.no_grad()
def get_embeddings(sequence: str) -> torch.Tensor:
    """Return per-residue embeddings, shape (L + 2, 960), including BOS/EOS."""
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    client = ESMC.from_pretrained("esmc_300m")
    client.eval()

    protein = ESMProtein(sequence=sequence)
    tensor = client.encode(protein)
    output = client.logits(
        tensor,
        LogitsConfig(sequence=True, return_embeddings=True),
    )
    # output.embeddings: (1, L + 2, 960) -> drop the batch dim
    return output.embeddings[0].float().cpu()


def main():
    parser = argparse.ArgumentParser(description="Generate ESMC-300M parity fixtures")
    parser.add_argument("--output", required=True, help="Output directory for safetensors")
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding reference sequence ({len(SEQUENCE)} residues) with ESMC-300M ...")
    embeddings = get_embeddings(SEQUENCE)
    print(f"  embeddings shape: {tuple(embeddings.shape)} (expected ({len(SEQUENCE) + 2}, 960))")

    out_path = output_dir / "esmc_parity.safetensors"
    save_file({"embeddings": embeddings.contiguous()}, str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
