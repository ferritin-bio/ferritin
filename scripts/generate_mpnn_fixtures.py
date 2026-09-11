#!/usr/bin/env python3
"""Generate ProteinMPNN and LigandMPNN parity fixtures.

Replaces `generate_proteinmpnn_fixtures.py`, which needed the *ProteinMPNN*
repo (`protein_mpnn_utils.tied_featurize`). Everything ferritin ports —
ProteinMPNN, LigandMPNN, SolubleMPNN — lives in the *LigandMPNN* repo, whose
`model_utils.py` is the file the Rust port is transcribed from, so that is the
reference this script runs.

What the fixture pins
---------------------
`score()` masks the decoder by a decoding order drawn from `randn`, so its
output is not reproducible across runs. The fixture instead pins the
**structure-only** forward pass: the decoder runs over the plain encoder
embeddings, every position seeing structure and no sequence. That is
deterministic, order-independent, and is exactly what
`ProteinMPNN::simple_decode` computes on the Rust side.

Alongside the log-probs it saves the featurizer and encoder outputs, so a
regression can be localised to a stage instead of only showing up as a
different distribution at the end.

Usage
-----
    git clone https://github.com/dauparas/LigandMPNN
    python -m venv .venv && .venv/bin/pip install prody torch safetensors numpy

    PYTHONPATH=LigandMPNN .venv/bin/python scripts/generate_mpnn_fixtures.py \
        --weights-dir crates/ferritin-test-data/data/ligandmpnn \
        --pdb crates/ferritin-test-data/data/structures/1BC8.pdb \
        --output crates/ferritin-plms/tests/fixtures/

1BC8 is the structure the Rust tests already use, and it carries a zinc ion —
so the same file exercises LigandMPNN's atom-context path rather than needing a
second structure for it.
"""

import argparse
from pathlib import Path

import torch

MODELS = {
    # fixture stem            (checkpoint stem,                model_type)
    "proteinmpnn_parity": ("proteinmpnn_v_48_020", "protein_mpnn"),
    "ligandmpnn_parity": ("ligandmpnn_v_32_020_25", "ligand_mpnn"),
}


def build(weights_path: Path, model_type: str, pdb: Path, device: str):
    import data_utils
    import model_utils

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    k_neighbors = checkpoint["num_edges"]
    atom_context_num = checkpoint.get("atom_context_num", 0)
    print(f"  {weights_path.name}: num_edges={k_neighbors} "
          f"atom_context_num={atom_context_num}")

    # augment_eps=0: the checkpoint's `noise_level` is the coordinate noise used
    # during TRAINING. Applying it here would make the fixture random.
    model = model_utils.ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        augment_eps=0.0,
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    protein, *_ = data_utils.parse_PDB(
        str(pdb),
        device=device,
        chains=[],
        parse_all_atoms=False,
        parse_atoms_with_zero_occupancy=False,
    )
    # run.py builds chain_mask outside parse_PDB; with no fixed or redesigned
    # residues every position is designable.
    protein["chain_mask"] = torch.ones_like(protein["mask"])

    features = data_utils.featurize(
        protein,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=atom_context_num,
        model_type=model_type,
    )

    out = {}
    h_V, h_E, E_idx = model.encode(features)
    # `enc_h_E` is (L, K, 128) — 2.3 MB for ProteinMPNN, far too large for a
    # committed fixture, and `enc_h_V` already localises a regression to the
    # encoder. `E_idx` is kept because a wrong neighbour list is the single
    # most likely featurizer break and is unmistakable when compared directly.
    out["enc_h_V"] = h_V[0].contiguous()
    out["E_idx"] = E_idx[0].to(torch.int32).contiguous()

    B, L = features["S"].shape
    h_S0 = torch.zeros((B, L, model.hidden_dim), device=device)
    h_EX = model_utils.cat_neighbors_nodes(h_S0, h_E, E_idx)
    h_EXV = model_utils.cat_neighbors_nodes(h_V, h_EX, E_idx)
    h = h_V
    for layer in model.decoder_layers:
        h = layer(h, h_EXV, features["mask"])
    logits = model.W_out(h)

    out["logits"] = logits[0].contiguous()
    out["log_probs"] = torch.log_softmax(logits, dim=-1)[0].contiguous()
    out["S"] = features["S"][0].to(torch.int32).contiguous()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from safetensors.torch import save_file

    torch.set_grad_enabled(False)
    weights_dir = Path(args.weights_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for stem, (checkpoint, model_type) in MODELS.items():
        print(f"{stem}:")
        tensors = build(weights_dir / f"{checkpoint}.pt", model_type,
                        Path(args.pdb), "cpu")
        for name, tensor in tensors.items():
            print(f"    {name:12} {tuple(tensor.shape)} {tensor.dtype}")
        path = output / f"{stem}.safetensors"
        save_file(tensors, str(path))
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
