#!/usr/bin/env sh
# Appends a Dolt-push step to .git/hooks/pre-push (ferritin-d2v).
#
# .git/hooks is not version-controlled, so a fresh clone (or a `bd` hook
# reinstall, which regenerates the beads-managed block above the markers
# this script looks for) has no guarantee that `git push` also pushes beads
# issue state. `bd hooks run pre-push` only runs backup/auto-export, and
# both skip themselves under BD_GIT_HOOK=1 — neither touches refs/dolt/data.
# Run this script once per clone (and again after any `bd` hook reinstall)
# to restore the explicit `bd dolt push` step.
set -eu

hook_file=".git/hooks/pre-push"
marker_begin="# --- BEGIN BEADS DOLT PUSH (ferritin-d2v) ---"
marker_end="# --- END BEADS DOLT PUSH (ferritin-d2v) ---"

if [ ! -d ".git" ]; then
  echo "error: run this from the repository root" >&2
  exit 1
fi

if [ -f "$hook_file" ] && grep -qF "$marker_begin" "$hook_file"; then
  echo "already installed: $hook_file"
  exit 0
fi

if [ ! -f "$hook_file" ]; then
  printf '#!/usr/bin/env sh\n' >"$hook_file"
  chmod +x "$hook_file"
fi

cat >>"$hook_file" <<'EOF'

# --- BEGIN BEADS DOLT PUSH (ferritin-d2v) ---
# The beads integration block above (if present) does NOT push Dolt data:
# 'bd hooks run pre-push' only runs the backup/auto-export hooks, and both
# skip themselves under BD_GIT_HOOK=1. refs/dolt/data is the only shared
# copy of issue state, so without an explicit push here, `git push` silently
# strands beads work on this machine while still reporting success.
# See ferritin-d2v.
if command -v bd >/dev/null 2>&1; then
  _bd_dolt_timeout=${BEADS_HOOK_TIMEOUT:-300}
  if command -v timeout >/dev/null 2>&1; then
    _bd_dolt_out=$(BD_GIT_HOOK=1 timeout "$_bd_dolt_timeout" bd dolt push 2>&1)
    _bd_dolt_exit=$?
  else
    _bd_dolt_out=$(BD_GIT_HOOK=1 bd dolt push 2>&1)
    _bd_dolt_exit=$?
  fi
  if [ $_bd_dolt_exit -ne 0 ]; then
    echo >&2 "=============================================================="
    echo >&2 "beads: 'bd dolt push' failed (exit $_bd_dolt_exit) — issue state"
    echo >&2 "may be stranded on this machine. Run 'bd dolt push' manually."
    echo >&2 "$_bd_dolt_out"
    echo >&2 "=============================================================="
  fi
  # Never block the code push on a beads sync failure.
fi
# --- END BEADS DOLT PUSH (ferritin-d2v) ---
EOF

chmod +x "$hook_file"
echo "installed: $hook_file"
