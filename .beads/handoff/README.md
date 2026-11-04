# Beads handoff from a cloud session

`bd dolt push` failed from the 2026-11-03 Claude Code cloud session: its git
proxy accepts pushes only to the session branch, so `refs/dolt/data` returned
HTTP 403. That session's Dolt database was discarded with its container. These
are the six issues it changed, exported with `bd export`:

| issue | change |
|---|---|
| ferritin-sxg | closed: AbLang2 ported and parity-verified |
| ferritin-y3b | new: AbLang2 ANARCI-aligned modes |
| ferritin-4dm | new: TCRLang-paired |
| ferritin-fju | note: HF CDN blocked in cloud sessions; ModernBERT-DNA config findings |
| ferritin-yjp, ferritin-u7g | note: same HF CDN block |

To apply on a machine that can push Dolt data:

```bash
bd import .beads/handoff/2026-11-03-cloud-session.jsonl
bd dolt push
```

Then delete this directory.
