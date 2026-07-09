---
name: brain-dumping
description: Create or update versioned HTML project memory memos from the current work session. Use when the user asks to brain dump, dump a session into memory, create a new memo, summarize today's work into project memory, reconcile current work with previous memos, or preserve agent/session artifacts such as SOUL.md, MEMORY.md, USER.md, SKILL.md files, scripts, prompts, tables, formulas, or figures in the memo record.
---

# Brain Dumping

## Purpose

Use this skill to turn the current session into a durable project memory memo.
The default action is to create the next versioned standalone HTML memo. Do not
create LaTeX or PDF memory memos unless the user explicitly asks for them.

## Workflow

1. Confirm the workspace root from the current working directory.
2. Find the memo folder by searching shallowly for common names:
   `Memory`, `Memory Memo`, `Memory-Memo`, `Memory_Memo`, `MemoryMemo`,
   `Memory Memos`, `Memos`, `Memo`, `Project Memo`, `Project Memos`,
   `Notes`, `Handoff`, `Context`, or `Context Memo`.
3. If no memo folder exists, create `Memory/` at the workspace root.
4. Identify existing versioned memo files, especially `memo_vNNN.html`.
   Select the next version number unless the user explicitly asks to append to
   or revise a particular memo.
5. Read the last two memo versions carefully enough to explain continuity,
   changed assumptions, new decisions, and corrections. Skim additional recent
   memos only if needed for context.
6. Inspect the current work products that the user names or that are clearly
   part of the session, using appropriate readers for `.html`, `.md`, `.docx`,
   `.msg`, code, images, and PDFs.
7. Create a standalone HTML memo following the existing project memo style when
   present.
   The memo must begin with a section summarizing what was conducted in the
   just-finished session.
8. Immediately after the session summary, add a reconciliation section that
   states what was changed, corrected, refined, or newly decided compared with
   the previous memory.
9. Include a continuity section explaining how the new memo connects to the
   last two memo versions.
10. Include concrete artifacts and paths where useful. Do not store secrets,
    tokens, passwords, private keys, or sensitive data in the memo.
11. Verify the `.html` memo exists, has nonzero size, opens as a complete HTML
    document, and references named artifacts accurately.
12. Do not create or retain LaTeX auxiliary files as part of memory dumping.

## Memo Content Requirements

Every newly created session memo should include:

- a title such as `Memo vNNN: <session topic>`;
- `Purpose of This Memo`;
- `Summary of This Session` as the first substantive section;
- `Reconciliation With Previous Memory`;
- `Connection to the Last Two Memos`;
- `Artifacts Created or Updated`;
- `Open Follow-Up Items`;
- `Memory Statement`.

It is normal for memos to be modified outside this skill. Do not assume every
memo section was created by this workflow. Later memos may contain multiple
session dumps, formulas, tables, graphs, figures, or appended notes. Preserve
that structure when extending an existing memo.

For standalone HTML memos or math notes that include formulas, write formulas
as LaTeX math notation and render them with MathJax when practical. Prefer
`\(...\)` for inline math and `\[...\]` for display equations. Do not encode
mathematical expressions primarily with HTML-only `<sub>`/`<sup>` markup unless
the note intentionally avoids JavaScript. This does not mean creating `.tex` or
PDF memo files; the default artifact remains standalone HTML.

## Editing Rules

- Prefer creating the next `memo_vNNN.html` rather than editing old memory.
- Edit an existing memo only when the user explicitly names it or the current
  task is clearly an append/revision.
- Use the project's current HTML conventions and visual style where possible.
- Keep the writing factual, dated, and traceable to session artifacts.
- When a user-provided document supersedes an agent draft, explicitly record
  that reconciliation.
