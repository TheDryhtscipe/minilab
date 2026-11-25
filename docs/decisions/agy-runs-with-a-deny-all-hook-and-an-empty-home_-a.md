---
id: doc-7w4
title: agy runs with a deny-all hook and an empty home, and each call must prove the
  hook fired
type: decision
scope: internal
created_at: '2026-11-25T00:38:54.120115Z'
updated_at: '2026-11-25T00:38:54.120115Z'
---

## Decision

The Gemini backend (`GeminiBackend` in `src/minilab/models/backends.py`, agy 1.5.16) blocks every agy tool with a `PreToolUse` hook. It also runs agy with an empty home folder, and it fails any call whose event stream does not prove that the hook denied every tool. With these, the isolation probe of 25 November 2026 (`runs/probes/2026-11-25.json`) marks Gemini `isolated`, so Gemini may generate in benchmark runs. Bead minilab-2be43f.

This updates the Gemini parts of doc-tvd. That doc says no deny rule covers agy's `search_web` tool, and it lists Gemini as "normal runs only".

### What each call does

1. `GeminiBackend.run` refuses to start (`ModelNotStarted`) if the working directory or any folder above it holds `.git`, `.jj`, or `.hg`. agy reads rules, skills, hooks, and MCP servers from every folder up to the repository root.
2. It writes `.agents/hooks.json` into the call's empty working directory. The hook matches every tool (`*`) and returns `deny`. Its reason holds a random nonce, new for each call.
3. It makes `.home` in the working directory. agy runs with `HOME` and the four XDG folders set inside it.
4. agy runs with `--output-format stream-json`, the session-only `AGY_SETTINGS`, and `--disable-slash-commands`.
5. `parse_agy` reads the stream and raises `ModelUnavailable` unless all of these hold:
   - There is one `init` event with a conversation id, and one `result` event.
   - Every step belongs to that conversation and has a step index.
   - Each `tool` step ends in state `ERROR`, with an error that starts `tool call denied by pre-tool hook:` and holds this call's nonce.
   - Each `subagent` step ends in state `ERROR`. agy reports no error text for these steps.
   - Every other step is `user_input`, `agent_response`, or `system_message`. Any other step type fails the call.
6. The reply's `tools_denied` counts the tool steps the hook denied. The probe passes a Gemini attempt only if that count is at least 1.

### Facts observed live on agy 1.5.16

- The hook blocked `search_web`, `read_file`, `run_command`, `run_workflow`, and `invoke_subagent`. A denied `invoke_subagent` or `run_workflow` starts no child agent.
- A hook that exits 1, times out, or prints text that is not JSON also stops the tool.
- An `allow` from another named hook, before or after ours, does not override our `deny`.
- Without the empty home, agy put the user's `~/.gemini/GEMINI.md` memory and the 27 skills in `~/.agents/skills` into every call. It also kept every prompt under `~/.gemini/antigravity-cli/`. With the empty home, agy still signs in through the OS keyring over D-Bus. It loads only its built-in skills and no user rules or memories. The home uses about 1.4 MB per call and is removed with the working directory.
- A `GEMINI.md` in the parent of a working directory outside any repository is not loaded.
- `@path` file mentions in the prompt load no file, and a prompt that starts with `/browser` is treated as text.

## Context

The probe of 24 November 2026 found that agy's settings deny file, URL, command, and MCP actions, but its built-in `search_web` tool still answered a web search. No settings key or deny rule covers `search_web`, the browser tools, or the subagent tools. The design (doc-gmj, "Blinding") leaves a tool out of both benchmark modes until it can be isolated.

## Options considered

- **Custom agent with a `tools:` list** (`.agents/agents/<name>.md`, selected with `--agent`). Rejected: agy reported "no custom agents are available" for an agent file in the temporary directory and in a trusted project folder, with either file layout. The global location is the user's config, so it was not tried.
- **A custom model entry with `modelFeatures.supportsToolCalls: false`.** Not tried: the hook works, and this needs a model entry in settings.
- **Adding `unsandboxed(*)` to the deny list.** Not tried: the hook already denies every tool.
- **Hook only, without the stream check.** Rejected by the user in favour of the hook plus the stream check, so that every benchmark call, not only the probe, shows the hook worked.

## Consequences

- A call in which Gemini tries a tool still succeeds if the hook denied it. Its tokens are charged as usual.
- A new step type in a later agy version fails every Gemini call until `_AGY_STEPS_WITHOUT_TOOLS` or the parser is updated. This is deliberate: an unknown step may run a tool.
- A tool that ran and then failed ends in `ERROR` too. The nonce check rejects it, because its error is not this call's hook denial.
- The prompt still goes on the command line (`--print=<prompt>`), so other local users can see it in the process list. `--input-format stream-json` may remove this; it was not tried.
- The probe cannot see whether Gemini's servers ground answers in search results without a tool call. In one red-team call, the model said it had no grounding. That rests on the model's own report.
