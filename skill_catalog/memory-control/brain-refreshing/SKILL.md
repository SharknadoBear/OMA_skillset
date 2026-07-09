---
name: brain-refreshing
description: Refresh Codex's working context before continuing a task. Use when the user mentions brain-refreshing, asks to refresh memory, resume work, reorient in the current workspace, or check project memos before acting. The workflow searches for Memory Memo folders or aliases, prioritizes HTML memory files and index pages, reads recent memos, browses the latest modified files in the working tree by name/timestamp, opens only a few high-signal recent files when there are many, cross-validates memo claims against live artifacts, diagnoses context drift, and returns to the active task.
---

# Brain Refreshing

## Purpose

Use this skill as a short orientation ritual before doing real work. Recover the user's HTML-based project memory, understand the current folder, diagnose context drift, and continue the task with less misalignment.

## Standing Workspace Objective Memos

When a discovered memo folder contains `memo_objective.html`, always read it
before reading recent session memos, even if it is not the newest memo. Treat it
as the standing objective for that workspace and keep it separate from the
chronological session record. If `memo_objective.html` is missing but
`memo_objective.tex` exists, use the TeX file only as a legacy fallback and
report that the HTML objective memory is missing.

## Hermes OMA Workspace Objective

When the workspace root is `Hermes_agent_fvcom_workflow`, always read
`Memory/memo_objective.html` before reading recent session memos. Treat it as
the standing objective for this folder: this workspace is for designing,
initiating, and smoke-testing Ocean Modeling Agent skills, while SkillOpt
optimization and Hermes Agent online deployment happen in separate sibling
workspaces. If this HTML objective memo is missing, fall back to
`Memory/memo_objective.tex` only when present, report that mismatch briefly,
and continue with the normal memo refresh.

## Workflow

1. Confirm the workspace root from the active working directory.
2. Search the workspace, starting shallow, for memo folders or aliases: `Memory Memo`, `Memory-Memo`, `Memory_Memo`, `MemoryMemo`, `Memory Memos`, `Memos`, `Memo`, `Project Memo`, `Project Memos`, `Memory`, `Notes`, `Handoff`, `Context`, or `Context Memo`.
3. If a discovered memo folder contains `memo_objective.html`, read it first, even if it is not the newest memo. If only `memo_objective.tex` exists, read it as a legacy fallback and call out that HTML objective memory is missing. In the `Hermes_agent_fvcom_workflow` workspace, apply the Hermes-specific objective note above after reading the file.
4. In memo folders, prioritize `.html` files by modification time, especially `index.html`, `memo_*.html`, and `math_notes_*.html`. Treat HTML as the user's normal memory format. Use `.tex` files only as legacy fallback when no relevant `.html` memo exists or when the `.html` file explicitly points to a missing legacy source. Use other readable formats such as `.md`, `.txt`, `.rst`, `.json`, `.yaml`, `.yml`, `.docx`, or `.pdf` only when no relevant `.html` memo exists or when they are clearly adjacent context.
5. Read the newest session memo carefully and skim one or two previous recent memos. Extract project state, open threads, decisions, warnings, next actions, and anything the user wanted remembered. Keep the standing objective from `memo_objective.html` separate from session chronology.
6. Browse the live working tree for recent changes before trusting the memo as current truth. List the newest modified files across the active workspace and important subtrees by name, timestamp, size, and path. When there are many recent files, open only a few high-signal readable files, usually 3--8 total: concise READMEs, manifests, summaries, small JSON/CSV reports, current scripts, tests, or logs. Prefer text artifacts that explain state over large binary outputs, generated figures, caches, or huge data files.
7. Cross-validate memo claims against live artifacts. Check whether the latest files show completed work, renamed workflows, newer analysis revisions, changed outputs, failed validations, or stale assumptions that the memo does not yet capture. Call out memo lag explicitly.
8. Diagnose context alignment: compare the recovered memo/workspace context with the most recent assistant output or current conversation direction. Call out omissions, wrong assumptions, stale framing, or task drift before continuing.
9. Inspect the present folder just enough to act well: top-level files, concise project docs, obvious entry points, `git status --short` if applicable, and `rg --files` for fast discovery.
10. Return to the active task with a compact refresh summary and the next action. Do not linger in recap mode.

## If Memos Are Missing

If no memo folder, alias, or relevant HTML memo is found, say so briefly and continue with workspace inspection. Do not block the task.

## Output Style

Keep the refresh concise: mention the latest `.html` memo or index read, previous memo skimmed, the latest live files inspected, any memo-versus-artifact mismatch, current project understanding, alignment diagnosis, and immediate next action. If there is no active task after refreshing, ask the user what to continue with.
