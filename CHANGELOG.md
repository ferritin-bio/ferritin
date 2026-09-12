# Changelog

Notable changes to the ferritin workspace. Newest first.

## Unreleased (0.4.0)

### Correctness notice — ProteinMPNN output before 0.4.0 is invalid

**If you used `ferritin-plms` ProteinMPNN at v0.3.3 or earlier, discard those
results and regenerate them.** This is not a precision regression or a tolerance
question. Measured against the LigandMPNN Python reference on 1BC8, ferritin's
ProteinMPNN agreed with the reference on **2 of 93 positions**. Chance agreement
over a 21-token vocabulary is about 4. The output was not approximately right —
it was unrelated to what the model computes.

Two defects were sufficient on their own:

- `crates/ferritin-plms/src/ligandmpnn/proteinfeaturesmodel.rs` read CB as the
  backbone oxygen out of atom37 slot 3, so the geometry fed to the model was
  wrong.
- `crates/ferritin-core/src/featurize/utilities.rs` sorted the neighbour search
  descending, so "nearest neighbours" were the farthest ones.

Both are present in the v0.3.3 tree (tagged 2025-04-09, commit `54a2209c`), so
**every tagged release that shipped ProteinMPNN shipped it broken**. Fixed on
main in `6ca31f16` (#212), which also wired up LigandMPNN; `proteinmpnn-v48-020`
and `ligandmpnn-v32-020-25` are now parity-verified against the reference.

Why nothing caught it: `test_pmpnn_parity_vs_python_reference` skipped on a
fixture nobody could generate, because `scripts/generate_proteinmpnn_fixtures.py`
required the `dauparas/ProteinMPNN` repo while the Rust is transcribed from
`dauparas/LigandMPNN`. The test passed vacuously for its entire existence, and
`ParityStatus::Unverified` read as "not yet checked" rather than "known to
disagree". See `ferritin-100.33`.

### Parity coverage

- ESMC-300M is now parity-verified against the EvolutionaryScale reference
  (cosine floor 0.999), closing the last family in the registry that had no
  verified member at all. `Family::Esmc` previously held three rows and no
  fixture, despite `scripts/generate_esmc_fixtures.py` needing only
  `pip install esm` to run.
- Every parity fixture declared in `PARITY_COVERAGE` is now generated and
  committed; the `NotGenerated` set is empty for the first time.
- New guard `test_every_family_has_a_verified_member` fails if any family with
  a loadable row has no verified member, so a whole architecture can no longer
  go unchecked while reading as merely "not checked" in the support matrix.
- SaProt-35M is now parity-verified against the HuggingFace reference (logit
  tolerance 1e-3). It shares `Family::Esm2` with the verified `esm2-t6-8m`, so
  the family guard already counted it as covered — but the half it does *not*
  share is the alphabet: two characters per residue over 446 tokens from a bare
  `vocab.txt`. Read that alphabet one character at a time and you still get a
  well-formed tensor, just with twice the rows and every other one `<unk>`,
  which is why it needed its own fixture (`ferritin-100.34`).
- `scripts/generate_esm2_fixtures.py` grew a `--variant saprot` mode, and now
  fails loudly if a sequence tokenizes to the wrong residue count or contains
  `<unk>` — a parity fixture over `<unk>` rows compares two models' opinions
  about nothing.

### Reading the support matrix

`not checked` now documents itself as "this output could be anything" rather
than "not known to be wrong". Nineteen registry rows remain `Unverified`.

Weigh such a row by how much it really shares with a verified one. A shared
`Family` tag is a claim about the *backbone*, not the whole path — SaProt proved
that by needing its own fixture despite sitting in an already-covered family.
`fastesm2-650`, `pepmlm-650m` and `dplm-650m` wear the `Esm2` tag with the same
caveat and remain unchecked (`dplm-650m` is a diffusion model, not a masked LM).
The rows where inherited trust is genuinely reasonable are the same-path ones:
the `esm2-t*` ladder, the five `esm1v` members, `esm1b`, `amplify-350m`,
`ankh-large`, `esmc-600m` and `esmc-6b`.

## v0.3.3 and earlier

See the git history. Note the ProteinMPNN correctness notice above, which
applies to all of these releases.
