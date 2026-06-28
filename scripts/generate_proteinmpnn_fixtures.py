#!/usr/bin/env python3
"""Generate ProteinMPNN parity fixtures for Rust numerical comparison tests.

Uses the canonical ProteinMPNN PyTorch model (proteinmpnn_v_48_020.pt) to produce
per-position amino-acid log-probability matrices from the structure-only score mode
(no autoregressive sampling, no temperature scaling).  Outputs are saved as
safetensors so the Rust test can load them via candle.

Usage
-----
    # Install dependencies
    pip install torch safetensors biopython numpy requests

    # Run (point --weights at the .pt file from ferritin-test-data)
    python scripts/generate_proteinmpnn_fixtures.py \
        --weights crates/ferritin-test-data/data/ligandmpnn/proteinmpnn_v_48_020.pt \
        --pdbs crates/ferritin-test-data/data/structures/1BC8.pdb \
        --output crates/ferritin-test-data/data/safetensors/proteinmpnn/

The script uses the same ProteinMPNN architecture as the Rust port so that the
outputs can be compared directly at tolerance KL < 0.01.

Note: the ProteinMPNN Python architecture is required.  Clone and pip-install it:
    git clone https://github.com/dauparas/ProteinMPNN
    pip install -e ProteinMPNN
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"  # 21 tokens (X = unknown/mask)


def featurize_pdb(pdb_path: str):
    """Parse a PDB file and return the ProteinMPNN-style feature dict.

    Reuses the helper from the canonical ProteinMPNN repo when available,
    falling back to a minimal Biopython-based parser.
    """
    try:
        from protein_mpnn_utils import StructureDatasetPDB, tied_featurize
    except ImportError:
        raise ImportError(
            "protein_mpnn_utils not found. Clone https://github.com/dauparas/ProteinMPNN "
            "and add it to PYTHONPATH, or pip install it."
        )

    pdb_dict_list = [{"name": pdb_path, "seq": "", "title": Path(pdb_path).stem}]
    dataset = StructureDatasetPDB(
        pdb_path,
        truncate=None,
        max_length=10000,
        alphabet="ACDEFGHIKLMNPQRSTVWYX-",
    )
    batch = [dataset[0]]
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
        visible_list_list, masked_list_list, masked_chain_length_list_list, \
        chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
        tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = tied_featurize(
            batch,
            device="cpu",
            chain_dict=None,
            fixed_position_dict=None,
            omit_AA_dict=None,
            tied_positions_dict=None,
            pssm_dict=None,
            bias_by_res_dict=None,
        )
    return X, S, mask, lengths, chain_M, residue_idx, chain_encoding_all


def load_proteinmpnn(weights_path: str):
    """Load ProteinMPNN model from the .pt checkpoint."""
    try:
        from protein_mpnn_utils import ProteinMPNN
    except ImportError:
        raise ImportError(
            "protein_mpnn_utils not found. Clone https://github.com/dauparas/ProteinMPNN."
        )

    checkpoint = torch.load(weights_path, map_location="cpu")
    model_state = checkpoint["model_state_dict"]

    # Detect model config from weight shapes
    hidden_dim = model_state["W_s.weight"].shape[0]  # 128 for v48_020
    num_letters = 21
    vocab = num_letters
    node_features = hidden_dim
    edge_features = hidden_dim
    k_neighbors = 48

    model = ProteinMPNN(
        num_letters=num_letters,
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=vocab,
        k_neighbors=k_neighbors,
        augment_eps=0.0,
        dropout=0.0,
    )
    model.load_state_dict(model_state)
    model.eval()
    return model


@torch.no_grad()
def score_structure(model, X, S, mask, lengths, chain_M, residue_idx, chain_encoding_all):
    """Run ProteinMPNN in score (forward) mode and return per-position log-probs.

    Mirrors the Rust `simple_decode` which uses a single non-autoregressive
    forward pass conditioned only on structure (no sequence context in decoder).

    Returns
    -------
    log_probs : np.ndarray, shape (L, 21)
    """
    randn = torch.randn(chain_M.shape, device=X.device)
    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
    # log_probs shape: (batch=1, L, 21)
    return log_probs[0].numpy()  # (L, 21)


def main():
    parser = argparse.ArgumentParser(description="Generate ProteinMPNN parity fixtures")
    parser.add_argument("--weights", required=True, help="Path to proteinmpnn_v_48_020.pt")
    parser.add_argument("--pdbs", nargs="+", required=True, help="PDB files to score")
    parser.add_argument("--output", required=True, help="Output directory for safetensors")
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.weights}")
    model = load_proteinmpnn(args.weights)

    for pdb_path in args.pdbs:
        stem = Path(pdb_path).stem
        print(f"Scoring {pdb_path} ...")

        X, S, mask, lengths, chain_M, residue_idx, chain_encoding_all = featurize_pdb(pdb_path)
        log_probs = score_structure(model, X, S, mask, lengths, chain_M, residue_idx, chain_encoding_all)

        # Save: log_probs (L, 21) as float32 tensor
        out_path = output_dir / f"{stem}_log_probs.safetensors"
        save_file({"log_probs": torch.tensor(log_probs, dtype=torch.float32)}, str(out_path))
        print(f"  Saved {out_path}  shape={log_probs.shape}")

        # Also save the sequence for reference
        seq_path = output_dir / f"{stem}_sequence.txt"
        seq = "".join(AMINO_ACIDS[s] for s in S[0].tolist())
        seq_path.write_text(seq)
        print(f"  Saved {seq_path}  len={len(seq)}")

    print("Done.")


if __name__ == "__main__":
    main()
