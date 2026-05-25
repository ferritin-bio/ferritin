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
