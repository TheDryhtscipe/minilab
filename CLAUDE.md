# Minilab

Minilab is a filter-first search loop for cross-domain ML mathematics. Read the design, DOCA doc-gmj ("Minilab design: a filter-first search loop for cross-domain ML mathematics"), before any task. Then work from the build plan, DOCA doc-5wh ("Minilab build plan"), one task per bead, with the `/bead` skill. If the plan and the design disagree, the design wins: flag the conflict on the bead.

## Global constraints

- Repo: `/home/thedryhtscipe/dev/sci/minilab`. Beads prefix `minilab`. Lane: focused.
- No `Any` types. `uv run pyright` in strict mode must pass on `src/` and `tests/`.
- Never write markdown files except README.md, AGENTS.md, and CLAUDE.md. Documentation goes in DOCA.
- Never read or change `~/dev/sci/experiments/crossconj`. Pilot files in `~/dev/sci/experiments/crossconj-ml` are read only.
- Run every model call isolated: an empty temporary working directory, no file tools, no web tools, and no MCP. For `claude -p`: no `--bare`, `--tools ""`, `--setting-sources ""`, `--strict-mcp-config`, `--no-session-persistence`, and `ANTHROPIC_*` stripped from the environment (MuninnDB `common` memory `01M3WPMQETWRFJQPDGFY0ZVBJG`).
- Model roles: Gemini generates only. A checker is never from the family that wrote the thing it checks. Codex budget goes to checking calls before generation.
- A kill needs a certificate from library code. A survival needs a coverage report.
- Benchmark constants: hidden results first posted on arXiv on or after `2026-07-01`; literature date cap `2026-06-30`; at most 8 domain pairs; memory policy `blind`.
- Tests never call a live model or a live web API by default. Mark live checks `@pytest.mark.live`; they run only with `-m live`.
- Commit messages state the change and may name the bead. No AI co-author lines.
- Remote: the public repo `TheDryhtscipe/minilab` on GitHub (use the TheDryhtscipe account). Push before you close a bead. The repo is public, so never commit anything from the sealed hidden set (plan Task 18).

## Commands

- Python: `uv sync`, then `uv run pytest`, `uv run pyright`, and `uv run ruff check`.
- Lean: in `lean/`, run `lake exe cache get`, then `lake build`. If `lake` is not on `PATH`, use `~/.elan/bin/lake`.
