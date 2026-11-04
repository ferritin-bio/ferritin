#!/usr/bin/env python3
"""Generate the AbLang2 parity fixture for the Rust tests (ferritin-sxg).

Runs the **official** AbLang2 code (oxpig/AbLang2, pinned below) on the
**official** paired-antibody weights (Zenodo record 10185169, pinned by
checksum) and saves token ids, hidden states, masked-LM logits and the
upstream scoring outputs as safetensors. `tests/test_plm_ablang2.rs` loads the
result and asserts the Candle port agrees.

What this pins down, because each is easy to get wrong in a port:

* **Input formatting.** `ablang2.pretrained.__call__` turns a `(heavy, light)`
  pair into `<HEAVY>|<LIGHT>` and then deletes any `<>`, so a heavy-only input
  is `<HEAVY>|` and a light-only one is `|<LIGHT>`. The fixture stores the ids
  that formatting produces, so the Rust tokenizer is checked against the
  official path, not against a re-derivation of it.
* **Residue alignment.** Upstream residue outputs keep the special-token rows
  (`res_to_list` only trims padding). Rows here are therefore *token* rows;
  the Rust test maps them to residues through its own alignment and checks
  that mapping separately.
* **Masked and unknown residues.** `*` is the mask token and `X` the unknown
  token *inside* chain text; both are covered.
* **Padding.** The batch case runs three unequal-length inputs through one
  padded forward pass, so a port that ignores the padding mask fails.
* **Scoring.** `pseudo_log_likelihood` and `confidence` are upstream's own
  numbers, computed by upstream's own code.

The upstream source is fetched file-by-file from raw.githubusercontent.com at
the pinned commit (an immutable URL), so no clone or pip install of AbLang2 is
needed, and the code that ran is exactly the code named in the fixture
metadata. Its runtime dependencies are `einops` and `rotary-embedding-torch`.

Usage
-----
    pip install torch safetensors numpy einops rotary-embedding-torch==0.9.1

    python scripts/generate_ablang2_fixtures.py \\
        --output crates/ferritin-plms/tests/fixtures/

Pass `--weights-dir` to reuse an already-extracted `ablang2-weights.tar.gz`
(it must hold `model.pt` and `hparams.json`); otherwise the archive is
downloaded from Zenodo and checked against its published MD5.

Regenerating under a different torch build moves the floats by ~1e-6. That is
noise, not a change in the reference, so leave a committed fixture alone
unless the inputs or the pinned revisions change.
"""

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

# ── Pinned provenance ────────────────────────────────────────────────────────

ABLANG2_REPO = "oxpig/AbLang2"
ABLANG2_COMMIT = "586af3083c32b5fb2f0a1c855f2ad4f1cad15ec3"
ABLANG2_FILES = [
    "ablang2/__init__.py",
    "ablang2/pretrained.py",
    "ablang2/load_model.py",
    "ablang2/models/__init__.py",
    "ablang2/models/ablang2/__init__.py",
    "ablang2/models/ablang2/ablang.py",
    "ablang2/models/ablang2/encoderblock.py",
    "ablang2/models/ablang2/tokenizers.py",
    "ablang2/models/ablang2/vocab.py",
    "ablang2/pretrained_utils/__init__.py",
    "ablang2/pretrained_utils/restoration.py",
    "ablang2/pretrained_utils/scores.py",
    "ablang2/pretrained_utils/encodings.py",
    "ablang2/pretrained_utils/extra_utils.py",
    "ablang2/pretrained_utils/alignment.py",
]

ZENODO_RECORD = "10185169"
ZENODO_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/ablang2-weights.tar.gz/content"
)
# Published by Zenodo for the archive.
ARCHIVE_MD5 = "6425b35e9b83750fde67a6a6d240d996"
# Of the extracted files, recorded here so a re-packaged archive is noticed.
MODEL_PT_SHA256 = "56d6f07862a6f824f88c8707bbc03e4026c9db762be2d3041e9767e2e6f86386"

# ── Inputs ───────────────────────────────────────────────────────────────────

# The VH/VL pair from the AbLang2 README.
VH = (
    "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTST"
    "AYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS"
)
VL = (
    "DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLR"
    "ISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK"
)


def masked(seq: str, positions) -> str:
    chars = list(seq)
    for p in positions:
        chars[p] = "*"
    return "".join(chars)


# name -> (heavy, light). Order matters only for readability.
CASES = {
    # Full-length pair: the realistic case, and the only one whose hidden
    # states are stored at full length.
    "paired": (VH, VL),
    # Single-chain inputs, which upstream formats asymmetrically.
    "heavy_only": (VH, ""),
    "light_only": ("", VL),
    # Short pair: cheap enough to store per-layer intermediates for.
    "short_pair": (VH[:30], VL[:25]),
    # Masked residues in both chains, including CDR-H3 and the first residue.
    "masked": (masked(VH, [0, 30, 99, 100, 101]), masked(VL, [5, 90])),
    # Unknown residue token inside chain text.
    "unknown": (VH[:20] + "X" + VH[21:40], "X" + VL[1:30]),
}

# One padded forward pass over unequal lengths.
BATCH = ["short_pair", "heavy_only", "unknown"]

PLL_CASES = ["short_pair", "unknown"]
CONFIDENCE_CASES = ["paired", "masked", "unknown"]

# Hidden states after these layers (0 = embedding) for `short_pair`.
LAYERS = [0, 1, 12]


def fetch_upstream(dest: Path) -> Path:
    base = f"https://raw.githubusercontent.com/{ABLANG2_REPO}/{ABLANG2_COMMIT}"
    for rel in ABLANG2_FILES:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(f"{base}/{rel}") as r:
                out.write_bytes(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404 and rel.endswith("__init__.py"):
                # Namespace package upstream; an empty file is equivalent.
                out.write_text("")
            else:
                raise
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(dest: Path) -> Path:
    archive = dest / "ablang2-weights.tar.gz"
    urllib.request.urlretrieve(ZENODO_URL, archive)
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    if md5 != ARCHIVE_MD5:
        sys.exit(f"archive md5 {md5} != published {ARCHIVE_MD5}")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    work = Path(tempfile.mkdtemp(prefix="ablang2-ref-"))
    src = fetch_upstream(work / "src")
    weights = args.weights_dir or fetch_weights(work / "weights")

    model_pt = weights / "model.pt"
    digest = sha256(model_pt)
    if digest != MODEL_PT_SHA256:
        sys.exit(f"model.pt sha256 {digest} != pinned {MODEL_PT_SHA256}")
    hparams = json.loads((weights / "hparams.json").read_text())

    sys.path.insert(0, str(src))
    import ablang2  # noqa: E402  (pinned upstream source)
    from ablang2.load_model import fetch_ablang2  # noqa: E402
    from ablang2.pretrained import add_extra_tokens, pretrained  # noqa: E402

    model, tokenizer, _ = fetch_ablang2(str(weights), device="cpu")
    model.eval()

    # Build the official `pretrained` front-end without its __init__, which
    # would download the weights a second time into site-packages.
    ab = pretrained.__new__(pretrained)
    ab.used_device = torch.device("cpu")
    ab.AbLang, ab.tokenizer, ab.hparams = model, tokenizer, hparams
    ab.AbRep = model.AbRep
    ab.ncpu = 1
    ab.spread = 11

    tensors = {}
    meta = {
        "ablang2_repo": ABLANG2_REPO,
        "ablang2_commit": ABLANG2_COMMIT,
        "zenodo_record": ZENODO_RECORD,
        "archive_md5": ARCHIVE_MD5,
        "model_pt_sha256": digest,
        "hparams": json.dumps(hparams, sort_keys=True),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "cases": json.dumps({k: list(v) for k, v in CASES.items()}),
        "batch": json.dumps(BATCH),
        "layers": json.dumps(LAYERS),
    }
    try:
        import rotary_embedding_torch

        meta["rotary_embedding_torch"] = getattr(rotary_embedding_torch, "__version__", "?")
    except Exception:  # pragma: no cover - recorded, not required
        pass

    with torch.no_grad():
        for name, pair in CASES.items():
            # Upstream's own formatting: `<H>|<L>` with empty chains dropped.
            text = add_extra_tokens(pair)
            ids = tokenizer([text], pad=True, w_extra_tkns=False)
            tensors[f"ids.{name}"] = ids[0].to(torch.int64).contiguous()

            # The official API, unaligned: one array per input, padding cut.
            logits = ab([pair[0], pair[1]], mode="likelihood")[0]
            tensors[f"logits.{name}"] = torch.from_numpy(np.asarray(logits, dtype=np.float32))
            if name in ("paired", "short_pair"):
                hidden = ab([pair[0], pair[1]], mode="rescoding")[0]
                tensors[f"hidden.{name}"] = torch.from_numpy(
                    np.asarray(hidden, dtype=np.float32)
                )

            assert tensors[f"logits.{name}"].shape[0] == len(text), name

        # Per-layer intermediates (pre final LayerNorm) for the short pair.
        short_ids = tensors["ids.short_pair"].unsqueeze(0)
        reps = model(short_ids, return_rep_layers=LAYERS)
        for n in LAYERS:
            tensors[f"layer{n}.short_pair"] = reps[n][0].contiguous()

        # Unequal-length padded batch through the official API.
        pairs = [CASES[n] for n in BATCH]
        batch_ids = tokenizer(
            [add_extra_tokens(p) for p in pairs],
            pad=True,
            w_extra_tkns=False,
        )
        tensors["batch.ids"] = batch_ids.to(torch.int64).contiguous()
        tensors["batch.logits"] = model(batch_ids).contiguous()

        # Upstream scoring, as returned by the official API.
        # PLL costs one forward pass per residue, so it is pinned on the two
        # short cases (one with an unknown residue in each chain, which must
        # be skipped); confidence is one pass, so it covers the long ones too.
        for name in PLL_CASES:
            pair = CASES[name]
            pll = ab([pair[0], pair[1]], mode="pseudo_log_likelihood")
            tensors[f"pll.{name}"] = torch.tensor(np.asarray(pll, dtype=np.float32)).reshape(1)
        for name in CONFIDENCE_CASES:
            pair = CASES[name]
            conf = ab([pair[0], pair[1]], mode="confidence")
            tensors[f"confidence.{name}"] = torch.tensor(
                np.asarray(conf, dtype=np.float32)
            ).reshape(1)

        # Masked-residue restoration, unaligned.
        restored = ab([CASES["masked"][0], CASES["masked"][1]], mode="restore")[0]
        restored = str(restored)
        meta["restore.masked"] = restored
        # Also as bytes, since safetensors metadata is not readable from candle.
        tensors["restore.masked"] = torch.tensor(list(restored.encode("ascii")), dtype=torch.uint8)

        # The checkpoint's rotary inverse frequencies, so the Rust port is
        # checked against the buffer actually used rather than a formula.
        sd = torch.load(model_pt, map_location="cpu", weights_only=True)
        tensors["rotary.inv_freq"] = sd[
            "AbRep.encoder_blocks.0.multihead_attention.rotary_emb.freqs"
        ].contiguous()

    from safetensors.torch import save_file

    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "ablang2_parity.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out), metadata=meta)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(tensors)} tensors)")
    for k in sorted(tensors):
        print(f"  {k}: {tuple(tensors[k].shape)} {tensors[k].dtype}")
    print(f"  restore.masked: {meta['restore.masked']}")


if __name__ == "__main__":
    main()
