---
id: doc-53t
title: 'Known-result filter: date cap, unchecked sources, and confirmed matches'
type: decision
scope: internal
created_at: '2026-11-25T13:21:08.693345Z'
updated_at: '2026-11-25T13:36:16.438552Z'
---

# Known-result filter: date cap, unchecked sources, and confirmed matches

Bead minilab-4e42a7 (build plan doc-5wh, Task 10; design doc-gmj, stage 4). Code: `src/minilab/literature/`.

## Decisions

1. **Mathlib takes no part in the status under a date cap.** Loogle searches the latest mathlib, and the repo's mathlib pin is dated 19 November 2026, after the benchmark cap of 30 June 2026. Neither can be capped. With a cap set, a mathlib match goes in the notes and never sets `known`. Whether mathlib was searched does not count either. A Loogle outage cannot make the result `unchecked`. A mathlib quote that fails the check cannot make it `unconfirmed`. Without a cap, mathlib counts like the other sources. The user chose this.
2. **The status follows a fixed order.** `known` comes first, then `unchecked`, then `unconfirmed`, then `not_found`. A confirmed match sets `known`, even if another source failed. If a source that counts failed (outage, rate limit, unreadable response), the result is `unchecked`. The same holds if the checker's verdict cannot be read. The stage then writes its log to `runs/<run_id>/literature/` but does not write the record, and raises `LiteratureUnchecked`, so the stage runs again on resume. A reply of an unknown shape counts as a failed source, never as "not found". So do a date the code cannot place against the cap and a redirect off the host list. The user chose this.
3. **A match counts only if code can confirm it.** The checker must name a candidate it was shown, quote a passage from that candidate's text, and say the passage states the same result or a stronger one. The text holds only what the source said. It never holds the title or the labels the prompt adds ("Abstract:", "From the PDF", "Question:", "Module:"). The PDF link is in the candidate's note. On arXiv and MathOverflow the quote needs at least 40 characters after normalising. For mathlib it may be the declaration's name or at least 12 characters of its text. Normalising applies Unicode NFKC, joins a word split by a hyphen at a line break, writes >=, <=, and != for their symbols, and collapses whitespace. A proof by `exact?` counts without the checker, because Lean checked it.
4. **A match whose quote fails the check is `unconfirmed`.** The checker may say `same` or `implies` for a candidate it was shown, with a quote that fails the check. Then the result is `unconfirmed`, unless a higher status applies. The stage is written with status `unconfirmed`, and the record's status stays as it was. The notes name the match that could not be confirmed, so the human queue sees it. The user chose this.
5. **The date cap is applied to every piece of text, titles included.** arXiv papers whose first version is after the cap are dropped even if the API returns them. For a paper revised after the cap, the feed's title and abstract belong to the late version. Both are withheld, and version 1's PDF is read. MathOverflow questions and answers created after the cap are dropped. A post edited after the cap loses its body. A question edited after the cap also loses its title, and so do its answers, which carry that title. The candidate's note says what was withheld and why.
6. **The checker writes the search queries.** One call by the checker family writes queries for each source; a second call judges the candidates. If the query reply cannot be read, queries are made from the record's objects. An arXiv query with nothing left after cleaning is logged as rejected and not sent.
7. **One HTTP client enforces the host list.** It reaches only `arxiv.org`, `export.arxiv.org`, `api.stackexchange.com`, and `loogle.lean-lang.org`. It uses HTTPS on the default port, with no user name or password in the URL. A URL the caller passes that fails this raises `HostNotAllowed`, which is a bug in the caller. A redirect that fails it is the server's doing. It raises `SourceUnavailable`, and the source is `unchecked`. Searches given no client share one, so calls to arXiv stay three seconds apart. `pdftotext` runs under bwrap with no network, and never without bwrap.

## Limits

- Normalising cannot tell a word split at a line break from a real hyphen at a line end. A quote of "continuous-time" fails against PDF text that breaks the line after the hyphen.
- Apart from normalising, the quote check is literal. A quote that the checker rewrites from garbled PDF mathematics fails it. The result is then `unconfirmed`, and the match goes to the human queue in the notes.
- The candidate limits cut some hits. Each query's log says which of its hits the checker was shown.
- `exact?` runs only when a Lean statement is passed in. In the pipeline the formaliser runs after this stage, so it does not run yet.
