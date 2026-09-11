#!/usr/bin/env python3
"""Build — and where possible run — every example in the workspace.

The old version of this script carried a hand-written list of 16 example
names. Six of them no longer existed, five current ones were missing, every
invocation hardcoded ``--features metal`` so it only worked on Apple silicon,
and it caught ``CalledProcessError``, printed it, and exited 0 regardless — so
it could not serve as a smoke gate even when the list was right
(ferritin-100.14).

Two things keep it from going stale again:

* The example list comes from ``cargo metadata``, not from this file, so a new
  ``[[example]]`` is picked up the moment it is added.
* ``POLICY`` below says what to *do* with each example, and a name that appears
  in one and not the other is a hard error. Adding an example without deciding
  whether it can run unattended fails loudly rather than being skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

# ── What can actually be run unattended ──────────────────────────────────────
#
# BUILD  the example needs a display, a GPU surface, a running viewer, or a
#        keypress, so compiling it is the whole smoke test.
# RUN    the example terminates on its own with no input.
#
# The PLM examples download model weights on first use — several GB for the
# 650M-parameter checkpoints — which is why running is opt-in behind
# ``--run``. Building is cheap and is what actually rots.
BUILD, RUN = "build", "run"

POLICY: dict[str, str] = {
    # Bevy: opens a window. These three self-exit, but still need a display
    # and a GPU adapter, so they are no more runnable in CI than the rest.
    "bevy_basic_ball_and_stick": BUILD,
    "bevy_basic_putty": BUILD,
    "bevy_basic_snapshot": BUILD,
    "bevy_basic_spheres": BUILD,
    "bevy_capture_representations": BUILD,
    "bevy_interactive_viewer": BUILD,
    "bevy_molviewspec_viewer": BUILD,
    "bevy_screenshot": BUILD,  # waits for the spacebar
    # Needs a Rerun viewer listening.
    "rerun": BUILD,
    # Protein language models: download weights, print, exit.
    "amplify": RUN,
    "esm2": RUN,
    "esm3": RUN,
    "esmc": RUN,
    "ligandmpnn": RUN,
    # Pure geometry, writes an SVG and exits.
    "simple": RUN,
    "simple_02": RUN,
}


def discover() -> list[tuple[str, str, list[str]]]:
    """Every example target in the workspace: (package, example, features)."""
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    examples = []
    for package in json.loads(out.stdout)["packages"]:
        for target in package["targets"]:
            if "example" in target["kind"]:
                examples.append(
                    (package["name"], target["name"], target.get("required-features") or [])
                )
    return sorted(examples, key=lambda e: (e[0], e[1]))


def check_policy(examples: list[tuple[str, str, list[str]]]) -> None:
    """A policy that does not match the manifest is the bug this script had."""
    found = {name for _, name, _ in examples}
    declared = set(POLICY)
    problems = []
    for name in sorted(found - declared):
        problems.append(
            f"  {name}: in the manifest but not in POLICY — decide whether it can "
            f"run unattended ({BUILD!r} or {RUN!r})"
        )
    for name in sorted(declared - found):
        problems.append(f"  {name}: in POLICY but no such example — delete the entry")
    if problems:
        print("run_examples.py: POLICY is out of sync with cargo metadata:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        action="store_true",
        help="also run the examples that terminate on their own (downloads model weights)",
    )
    parser.add_argument(
        "--metal",
        action="store_true",
        help="add --features metal (Apple silicon only; off by default)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a failure instead of stopping at the first one",
    )
    parser.add_argument(
        "--list", action="store_true", help="print what would be built and run, then exit"
    )
    args = parser.parse_args()

    examples = discover()
    check_policy(examples)

    if args.list:
        for package, name, features in examples:
            action = POLICY[name]
            if action == RUN and not args.run:
                action = f"{BUILD} (run with --run)"
            feature_note = f" [+{','.join(features)}]" if features else ""
            print(f"{action:>20}  {package}::{name}{feature_note}")
        return 0

    failures: list[str] = []
    for package, name, features in examples:
        verb = "run" if POLICY[name] == RUN and args.run else "build"
        if args.metal:
            features = [*features, "metal"]

        cmd = ["cargo", verb, "-p", package, "--example", name]
        if features:
            cmd += ["--features", ",".join(features)]

        print(f"\n{'=' * 60}\n{verb}: {package}::{name}\n{'=' * 60}\n", flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as err:
            print(f"FAILED ({verb}): {package}::{name} — exit {err.returncode}", file=sys.stderr)
            failures.append(f"{package}::{name}")
            if not args.keep_going:
                break
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            return 130

    if failures:
        print(f"\n{len(failures)} example(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"\nAll {len(examples)} examples OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
