---
id: doc-53t
title: 'Known-result filter: date cap, unchecked sources, and confirmed matches'
type: decision
scope: internal
created_at: '2026-11-25T13:21:08.693345Z'
updated_at: '2026-11-25T13:21:08.693345Z'
---

# Known-result filter: date cap, unchecked sources, and confirmed matches

Bead minilab-4e42a7 (build plan doc-5wh, Task 10; design doc-gmj, stage 4). Code: `src/minilab/literature/`.

## Decisions

1. **Mathlib matches are notes only under a date cap.** Loogle searches the latest mathlib, and the repo's mathlib pin is dated 19 November 2026, after the benchmark cap of 30 June 2026. Neither can be capped. With a cap set, a mathlib match goes in the notes and never sets `known`. Without a cap it sets `known` as normal. The user chose this.
2. **An unchecked source leaves the record unwritten.** If a source fails (outage, rate limit, unreadable response) and no confirmed match came from the sources that answered, the result is `unchecked`. The stage writes its log to `runs/<run_id>/literature/` but does not write the record, and raises `LiteratureUnchecked`, so the stage runs again on resume. A confirmed match from a source that answered still sets `known`. An unreadable checker verdict is treated the same way. The user chose this.
3. **A match counts only if code can confirm it.** The checker must name a candidate it was shown, quote a passage that appears in that candidate's text (after Unicode and whitespace normalisation, at least 12 characters), and say the passage states the same result or a stronger one. A proof by `exact?` counts without the checker, because Lean checked it.
4. **The date cap is applied to every piece of text.** arXiv papers whose first version is after the cap are dropped even if the API returns them. For a paper revised after the cap, the feed's abstract is left out and version 1's PDF is read. MathOverflow questions and answers created after the cap are dropped; those edited after the cap keep their title but not their body.
5. **The checker writes the search queries.** One call by the checker family writes queries for each source; a second call judges the candidates. If the query reply cannot be read, queries are made from the record's objects.
6. **One HTTP client enforces the host list.** Only `arxiv.org`, `export.arxiv.org`, `api.stackexchange.com`, and `loogle.lean-lang.org`, over HTTPS, including after redirects. Calls to arXiv wait three seconds apart. `pdftotext` runs under bwrap with no network.

## Limits

- Paper titles from the arXiv feed belong to the latest version, even when it is after the cap.
- The quote check is literal. A quote that the checker rewrites from garbled PDF mathematics fails it, so the record is not `known`; the unverified match stays in the notes for the human queue.
- `exact?` runs only when a Lean statement is passed in. In the pipeline the formaliser runs after this stage, so it does not run yet.
