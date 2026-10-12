---
name: writing-style
description: "Rules for every text written in this project: code comments, docstrings, markdown files, READMEs, Confluence pages, Jira tickets, PR descriptions, and reports. Easy English for non-native readers: common words, no figurative language, no AI-sounding prose, no history narration, short titles. Use proactively whenever writing or editing any text, and when the user invokes /writing-style to check existing text."
argument-hint: "[optional: file, directory, or page to check]"
---

# writing-style

These rules apply to every text written in this project: code comments, docstrings, *.md files, READMEs, Confluence pages, Jira tickets, PR descriptions, and reports.

The goal: a reader who is not a native English speaker reads the text once, from top to bottom, and understands every sentence without stopping. Aim at a 16-year-old non-native reader: simple but precise, always with facts. Every rule below serves that goal. When two rules conflict, pick what is easier to read.

## Easy English

- *Always pick the most common word.* "use", not "utilize" or "leverage". "start", not "initiate". "also", not "furthermore". "about", not "approximately". "enough", not "sufficient". "smallest", not "mildest". "the same as", not "indistinguishable from". If a more common synonym says the same thing, the harder word is wrong.
- *No figurative language.* No metaphors, no idioms, no images, including the soft ones that feel natural to a native speaker: "yardstick", "frame", "landscape", "journey", "under the hood", "low-hanging fruit", "the price of", "goes stale", "steps in", "clears the bar", "fingerprints". Test: if the sentence describes a physical object or action that is not physically there, replace it with the literal fact.
  - Bad: "Two yardsticks frame every number."
  - Good: "Every model is compared against two values."
- *Project terms are not figurative language.* A term the project defines (floor, ceiling, anchor, fold, feature world) stays, spelled the same every time, and is explained in plain words on first use.
- *Simple never means vague.* Keep exact names, paths, numbers, and behavior. Precision comes from facts, not from big words.
- *Write complete, connected sentences.* Long enough that the reader can read continuously, joined with "so", "because", "and", "but". Not telegraphic fragments ("Retries: 3. On fail: raise.") and not 40-word chains either. One idea per sentence, and split a sentence that passes about 30 words.
- *Say only what the reader needs.* Cut whole details that do not change what the reader knows or does. Shorten by dropping content, never by compressing sentences into fragments.
- *Pick one voice per document.* Explanatory pages and proposals use "we". Specs and procedures are impersonal ("the model", "the job") with bare imperatives for steps. Do not mix the two in one document.
- *Use -, never the long dash —.*
- *No semicolons in prose.* Split into two sentences, or join with "so", "and" or "but". (Semicolons inside code are fine.)
- *For an aside inside a sentence, use parentheses.* Not paired hyphens ("- like this -").

## No AI-sounding text

Readers recognize generated text by its shape. Remove the shape:

- *Open with the plain fact, not a flourish.* No dramatic first sentences, no scene-setting, no sentence that sounds like a quote.
- *No rhetorical patterns built for effect.* No two-sentence reversals ("The answer isn't X. It's Y."), no negative listing ("Not a tool. Not a library. A platform."), no rhetorical questions, no dramatic short sentence for punch, no three-part rhythm added for effect ("fast, simple, and reliable"). A list of three real items is fine, but a triad built for sound is not.
- *No praise or drama about the work itself.* No "powerful", "elegant", "robust" (as praise), "seamless", "comprehensive", "crucial", "delve", "key" as an adjective.
- *No filler.* No "It is worth noting that", "In order to", "Note that", "Additionally", "However" as a paragraph opener, "At its core", "The reality is". Start with the fact.
- *No empty intensifiers.* "really", "actually", "simply", "truly", "genuinely", "very". An adverb that carries a fact ("roughly 2M rows", "deliberately not fitted") stays.
- *No summary sentences that restate what was just said.*
- *Do not vary a word for elegance.* The same object gets the same name in every sentence. A segment stays "segment".
- *No bold for emphasis inside prose, no exclamation marks, no emoji.* Bold marks a defined term on first use, a label, or a table header.

The /stop-slop skill holds the full pattern catalog: banned filler phrases in references/phrases.md and banned structures (binary contrasts, negative listing, dramatic fragmentation, false agency) in references/structures.md. Use it as a checklist on any prose. Three of its rules are relaxed here:

1. Passive voice and impersonal subjects are normal in technical documentation ("the model is trained on the other seven windows", "the row is flagged").
2. A short in-sentence contrast that closes a likely misreading ("a target, not a hard cap") is allowed. Only the dramatic two-sentence reversal is banned.
3. The essay voice ("you in the room", a human subject in every sentence) does not apply to technical documents.

## Titles

- Short noun phrases, one to four words: "Judging rule", "Overview", "Model cards", "Sources".
- *Never start a title with "The".* "Judging rule", not "The judging rule". The rare exception is a title that becomes wrong without it.
- No questions, no full sentences, no cleverness. Sentence case.

## Facts, not history

Text states what is true now. It never tells the story of how the text, the code, or the page came to be.

- No "this was changed because", "after we found", "previously X, now Y", "as discussed", "in the last task".
- Never mention the task, session, conversation, or tool that produced the text. A future reader gets nothing from knowing that a page was written while some other work was going on.
- The story of a change belongs in the commit message and the PR, never in files or pages.
- Do not record who made a decision, when, or the process around it. No "X decided", no "this overrides the earlier plan", no review or board framing. State the decided fact itself.
- Never narrate that something was added, then removed, then added again. A document states only the final design, as if it had always been that way.
- *Describe what is there, not what is absent.* Do not mention parts the subject does not use, and do not explain why something was excluded, removed, or retired. A thing that is not part of the subject does not belong in its text. The exclusion reason lives in the experiment record, the commit, or the PR.
  - Bad: "The reporter count nRepSeg is stored but not fitted: its value identifies the market." (The column is not a feature of the model, so the model's card must not mention it.)
  - Good: nothing. List only the features the model uses.
  - Exception: when a document compares items along a fixed field that every item answers, "none" is a fact, not noise ("Probe features: None").
- A past event can itself be a fact the reader needs. State it as a dated fact, not as a story. "The config declares moe-v2, but the published numbers were scored under moe-v1." Not: "we later bumped the config to moe-v2".
- *When moving a document from a draft space (Confluence, an RFC) into the repo or a site, verify every claim against the code first.* Keep only what is implemented, with the implementation's numbers, and delete proposals, alternatives, retired designs, superseded parameters, next-steps sections and approval furniture. An approved source does not make a copied claim true today: the code is the authority, the document follows it.

| Rule | Bad | Write instead |
|---|---|---|
| Do not restate the code. | # increment the counter above counter += 1 | Nothing. Delete it. |
| Do not narrate what you learned while coding. | # Use a set - after debugging we found duplicates crashed the run. | # Input ids can repeat, so keep only the first one. |
| No ticket IDs in comments or docs. | # Workaround for OMA-4521. | State the actual constraint, or nothing. The ticket goes in the branch name and the PR. |
| No numbers or names from the bug you just fixed. | # Timeout is 30s because the March run took 25s. | # The upstream service can be slower than the default timeout. |
| Do not guess what future callers might want. | # Callers may want to pass their own client later. | Nothing. Stick to facts about the code as it is. |

Exception: plan files under <area>/plans/ link their Jira ticket by design.

## When to write at all (code comments and docstrings)

Default to no comment. If the code, the function name, or the type signature already says it, do not repeat it in words.

Before keeping a comment, ask: would removing this confuse a reader who knows the language and can see the signature? If no, delete it.

A comment earns its place only when it states something the reader cannot see in the code:

- A hidden constraint. # The API rejects batches over 500 rows.
- A non-obvious invariant. # Rows arrive sorted by timestamp, and the loop depends on that.
- A real reason the simple version does not work. # strict mypy needs the defining-module import

Docstrings follow the same test: say something the name does not already say. Skip """Compute the score.""" on def compute_score. The structure rules (Args, Returns, Raises) are in CLAUDE.md.

## Checking existing text (/writing-style [path])

When the user invokes the skill on a file, directory, or page:

1. Read the text in scope.
2. Flag each violation with the rule it breaks.
3. Propose the fix. Often that is deletion, and sometimes a shorter factual rewrite.
4. Apply the fixes only after the user confirms.
