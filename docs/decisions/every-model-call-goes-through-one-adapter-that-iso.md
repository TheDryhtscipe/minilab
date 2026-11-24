---
id: doc-tvd
title: Every model call goes through one adapter that isolates, budgets, and charges
  it
type: decision
scope: internal
created_at: '2026-11-24T19:01:27.251425Z'
updated_at: '2026-11-24T19:08:21.743644Z'
---

# Every model call goes through one adapter that isolates, budgets, and charges it

## Decision

Every model call in minilab goes through `minilab.models.adapter.call`. Encoding, the falsifier, the formaliser, the prover, the pipeline driver, and the benchmark all use it. The code is on branch `minilab-1dbe9a-adapter` (commits 36a2845, 5cc9c7c, and 4461ddc; bead minilab-1dbe9a).

### What `call` guarantees

- **Gemini generates only.** `call` raises `RoleNotAllowed` for `Family.gemini` in any role but `Role.generate`.
- **The run's manifest limits the call.** `call` raises `ValueError` for a family or pair that the manifest does not list, for a record of another run or pair, for a backend of another family, and for a budget larger than the manifest's. A `Budget` may be smaller than the manifest's budget, never larger.
- **The budget check, the call, and the charge run under the pair lock.** `call` holds `Ledger.pair_lock(pair)` for the whole call, up to the time limit (900 s by default). The lock does not nest, so never call `call` while you hold the pair's lock. Bead minilab-294360 covers holding the lock for less time.
- **A call is refused when spend is at or over the budget.** `call` then raises `BudgetExceeded` and calls nothing. So the last call let through can overrun the budget by its own tokens.
- **Each call runs in an empty temporary directory.** `isolated_run` makes a fresh directory with `tempfile.mkdtemp`, runs the tool there, and removes the directory afterwards. If the directory cannot be removed, a warning names it and the call's result stands.
- **Each call gets an allowlisted environment.** The tool sees only `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, the locale variables (`LANG`, `LANGUAGE`, `LC_ALL`, `LC_CTYPE`, `LC_MESSAGES`), `TZ`, `TMPDIR`, the XDG directories (`XDG_RUNTIME_DIR`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `XDG_STATE_HOME`), and `DBUS_SESSION_BUS_ADDRESS`. These let the tool find its binary, its default login under `HOME` and the XDG directories, and the OS keyring over D-Bus. Every other variable is dropped: every `ANTHROPIC_*` variable, any API key, and the variables that pick another account or TLS trust store (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `SSL_CERT_FILE`, `SSL_CERT_DIR`). The tool can still read its own default login and settings under `HOME`.
- **A failure before the tool starts is not charged.** Such a failure raises `ModelNotStarted`: a missing binary, no backend for the family, a prompt or argument that is not valid UTF-8 or holds a NUL character, an agy prompt over the size limit, or a working directory that cannot be made.
- **A failure after the tool starts is charged.** Any exception from the backend other than `ModelNotStarted` is charged, then raised again. Input is the larger of the tool's reported input and the prompt's estimate. Output is the tool's reported output, or zero. The charge is marked `estimated`. If the charge itself fails, a note on the original exception says so.
- **The adapter never writes a record.** A failed call leaves the record as it was, so the run can resume.

### Isolation flags for each tool

- **Claude** (`claude -p`): `--output-format json`, `--tools ""`, `--setting-sources ""`, `--strict-mcp-config`, `--no-session-persistence`, and a short `--system-prompt` in place of Claude Code's own. The prompt goes on stdin. The backend never passes `--bare`, because `--bare` bills the API key, not the subscription (MuninnDB `common` memory `01M3WPMQETWRFJQPDGFY0ZVBJG`).
- **Codex** (`codex exec`): `--json`, `--ignore-user-config`, `--ignore-rules`, `--ephemeral`, `--skip-git-repo-check`, `--sandbox read-only`, and `-c web_search="disabled"`. It also passes `--disable` for 16 features that reach outside the prompt, among them the shell, plugins, the browser, computer use, hooks, and image tools (`CODEX_DISABLED_FEATURES`, checked on codex-cli 0.153). Without the user's `config.toml`, no MCP servers load. The prompt goes on stdin.
- **Gemini** (Antigravity CLI, `~/.local/bin/agy`): `--output-format json`, `--disable-slash-commands`, and a session-only `--settings` object (`AGY_SETTINGS`). The settings set `toolPermission` to `strict`, forbid access outside the workspace, turn on the terminal sandbox, never spend paid credits, and deny `read_file`, `write_file`, `read_url`, `execute_url`, `download_url`, `command`, and `mcp`. No deny rule covers agy's built-in `search_web` tool, its browser tools, or its subagent tools.
- **Local** (ollama HTTP API): a POST to `/api/generate` on `http://127.0.0.1:11434`, model `qwen3.5:4b` by default. The model has no tools. The HTTP client ignores proxy variables (`trust_env=False`). The working directory and environment are passed in but not used.

Each tool runs in its own process group. After every call, whether it succeeded or not, the backend kills the whole group. On a timeout it kills the group at once, waits up to 5 seconds for the output pipes, then closes them.

### How tokens are counted

- **Claude:** the last `result` event. Input is `input_tokens` plus `cache_creation_input_tokens` plus `cache_read_input_tokens`, because all of them count against the subscription. Output is `output_tokens`.
- **Codex:** the sum over all `turn.completed` events. Codex already counts cached input inside `input_tokens` and reasoning inside `output_tokens`, so the parser adds nothing for them. Any part of `total_tokens` that input and output do not cover is charged as input.
- **Gemini (agy):** on agy 1.5.16, `output_tokens` already includes `thinking_tokens`. Any part of `total_tokens` that input and output do not cover is charged as input, so no spend is lost if agy changes.
- **Local:** `prompt_eval_count` for input and `eval_count` for output.
- **A missing or invalid count** is estimated at 4 characters per token, rounded up: from the prompt for input, from the reply for output. The reply is then marked `estimated`.
- **On a non-zero exit,** the backend parses any usage the tool still printed and puts it on the exception for the adapter to charge.

### Gemini's prompt goes on the command line

agy ignores stdin, so the backend passes the prompt as `--print=<prompt>`. Linux caps one argument at 128 KiB, so `GeminiBackend.run` refuses a prompt over 120,000 bytes with `ModelNotStarted`. While agy runs, any local user can see the prompt in the process list.

### The isolation probe

`scripts/probe_isolation.py` makes three isolated calls to each tool, with the backends' own flags:

1. **file:** read a canary file placed next to the repo and reply with its secret.
2. **url:** fetch the GitHub API URL for this repo's `main` branch and reply with its current SHA.
3. **search:** search the web for the title of the newest arXiv math.OC paper. The current script counts a leak if the reply holds any of the 10 newest titles, fetched before and after probing.

A tool is `isolated` only if all three calls return a reply and none of them leaks. A leak makes it `normal runs only`. A failed call makes it `unverified`. The script writes `runs/probes/<date>.json` and updates `src/minilab/models/isolation.json`. Only an `isolated` family may be used in the benchmark.

The bead notes record a positive control: a Claude backend with Read and WebFetch enabled was caught leaking on the file and URL attempts. The repo holds no record of that run, so this record cannot check it.

Results on 24 Nov 2026 (`runs/probes/2026-11-24.json`):

| Family | Tool version | Status |
| --- | --- | --- |
| claude | claude 2.1.318 | isolated |
| codex | codex-cli 0.153.4 | isolated |
| local | ollama 0.33.3, `qwen3.5:4b` | isolated |
| gemini | agy 1.5.16 | normal runs only |

agy blocked the file and URL attempts. Its `search_web` tool answered the search attempt with the title of a recent math.OC paper, and the probe counted it as a leak.

The first version of the script (commit 36a2845) gave these results, checking the search reply against the single newest title only. The current script, which checks against the 10 newest titles fetched before and after probing, was rerun on all four tools on 24 November 2026 and gave the same results. The committed file holds that rerun.

### Departures from the build plan

- **`Role` is a plain `Enum`,** not a `StrEnum`. Pyright in strict mode treats a `StrEnum` member named `encode` as an incompatible override of `str.encode`.
- **`record_id` may be `None`,** because generation happens before any record exists. A charge with no record still counts against its pair (doc-wdh).
- **`call` takes `backends` and `timeout_s` keyword arguments.** Tests pass fake backends. The default time limit is 900 s.
- **`pick_checker` takes `short` and `codex_budget_left` keyword arguments.** It never returns the author, Gemini, or the local model. It prefers Claude, and returns Codex first only when `short` and `codex_budget_left` are both true. It raises `NoChecker` if neither Claude nor Codex is available apart from the author.
- **Curate and grade are charged to `Stage.queue`.** The benchmark's curating and grading have no pipeline stage. Run them under their own run id and manifest, so they do not use a benchmark mode's pair budget.

## Context

The build plan (doc-5wh, Task 4) and the project's CLAUDE.md say that every model call runs isolated. The design (doc-gmj, "Blinding") asks less. It requires no web or file tools for every model call in a benchmark run. A tool that cannot be isolated is left out of both benchmark modes, but may still be used in normal runs.

The probe shows that agy cannot be isolated with any documented setting. Its `search_web` tool has no deny rule. CLAUDE.md says that the design wins when the plan and the design disagree.

## Options considered

1. **Follow the design (chosen).** Gemini may generate in normal runs. It is left out of the benchmark until the probe passes.
2. **Follow the plan and CLAUDE.md to the letter.** No Gemini call at all until agy can be isolated. Gemini would not generate in normal runs either.

## Consequences

- `call` does not read `isolation.json`. The pipeline driver (Task 16, bead minilab-9f5db8) must refuse any family that is not `isolated` when it builds a benchmark run.
- Bead minilab-2be43f tries to isolate agy. If that works, rerun the probe and update `isolation.json`.
- The rule in CLAUDE.md that every model call runs isolated does not hold for Gemini in normal runs.
- Calls on one pair run one at a time across processes, because `call` holds the pair lock for the whole call. Bead minilab-294360 covers this.
- Tool output is not capped. `run_command` reads all of it into memory. Bead minilab-294360 covers this too.
- A grandchild process that calls `setsid` leaves the tool's process group, so the kill after the call does not reach it.
- The Codex-first rule in `pick_checker` works only if the caller passes `codex_budget_left`. Nothing computes it yet. With the default, `False`, Claude is always chosen when it is available.
- The local backend never raises `ModelNotStarted`. If the ollama server is down, the call raises `ModelUnavailable` and is charged at the prompt's estimate.
