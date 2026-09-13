# Ferritin Build and Development Guide

## Build Commands
- Build specific packages: `cargo build -p ferritin-core -p ferritin-pymol -p ferritin-bevy`
- Run examples: `cargo run --example <example_name>` (e.g. `cargo run --example amplify`)
- Build with metal feature: `cargo run --example <example_name> --features metal`

## Test Commands
- Run all tests: `cargo test`
- Run tests with ignored tests: `cargo test -- --include-ignored`
- Run specific test: `cargo test <test_name>`
- Run specific test with output: `cargo test <test_name> -- --nocapture`
- Run specific package test: `cargo test -p <package_name>`

## Documentation and Utilities
- Generate docs: `cargo doc --workspace --no-deps`
- Clean project: `cargo clean -p ferritin-core -p ferritin-pymol -p ferritin-bevy`

## Code Style Guidelines
- **Naming**: Use snake_case for functions/variables, PascalCase for types/structs
- **Modules**: Organize features into modules with clear separation of concerns
- **Imports**: Group imports logically (std, external crates, internal modules)
- **Error Handling**: Use Result types with descriptive error messages
- **Testing**: Write unit tests with descriptive names prefixed with `test_`
- **Documentation**: Include module-level and function-level documentation
- **Types**: Use strong typing and appropriate enums for representing states

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## Beads Durability (ferritin-d2v)

**`git push` does NOT push beads data.** The two are entirely separate transports,
and nothing warns you when only one of them runs.

Issue state lives in a local Dolt DB whose only shared copy is `refs/dolt/data` on
the git remote. `.beads/.gitignore` excludes `embeddeddolt/` and `export.auto` is
`false`, so there is no JSONL fallback committed to the repo either. The installed
`.git/hooks/pre-push` runs `bd hooks run pre-push`, which does exactly two things —
backup and auto-export — and **both log `skipping — running as git hook` and exit 0**.
So a `git push` pushes no Dolt data, and reports success either way. That is how two
days of issue work once lived on one laptop only.

**Therefore, step 4 of Session Completion above is incomplete on its own.** Whenever a
session has touched beads at all (`bd create`, `bd update`, `bd close`, `bd note`,
`bd remember`), it must also run:

```bash
bd dolt push
```

It must print `Push complete.` and exit 0. It is idempotent and takes ~3s, so run it
even when unsure. A session that pushed code but not Dolt has stranded its issue work
exactly as invisibly as before.

To check at session START whether a previous session stranded anything, read the
remote ref and see whether its date matches the last known beads activity:

```bash
git ls-remote origin refs/dolt/data
```

Two known display quirks, neither of them a fault: `bd dolt show` reports
`Remotes: (none)` while `bd dolt remote list` correctly reports `origin` — the two
views disagree, and `remote list` is the accurate one. And `sync.remote` in
`.beads/config.yaml` is a `git+https://` URL while `origin` is SSH; this is
deliberate (see the comment there) and does not impede the push.

### Local safety net: pre-push hook

`.git/hooks` is not version-controlled, so it cannot durably fix this for every
clone — but as a per-machine backstop, `scripts/install-git-hooks.sh` appends a
`bd dolt push` step to `.git/hooks/pre-push` (after the beads-managed markers, so
a `bd` hook reinstall won't silently remove it — though it will need re-running).
It warns loudly on failure but never blocks the code push. Run it once per clone:

```bash
sh scripts/install-git-hooks.sh
```

This does not replace the manual `bd dolt push` in Session Completion above —
it is a backstop for the case where that step gets forgotten, not a substitute
for it.
