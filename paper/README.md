# Paper sources

Load this folder into Overleaf (**New Project → Upload Project**, zip the `paper/` directory).
Compiler **pdfLaTeX**, bibliography **BibTeX**. `IEEEtran` is preinstalled on Overleaf.

| File | What it is |
| --- | --- |
| `main.tex` | the paper; `\documentclass[conference]{IEEEtran}` |
| `references.bib` | reference list, rebuilt from scratch — see the header comment for the verification rule |
| `figures/` | generated figures; anything absent renders as a labelled placeholder box |

## Draft markers

- `\todo{...}` — red, work still owed
- `\verifycite{}` — orange superscript, a citation whose record or attributed claim is unconfirmed

Both are defined in `main.tex` and are meant to be deleted, not left in a submission.

## Rules this draft follows

- Every number is read from a committed manifest in `../artifacts/`, never transcribed from memory
- Research gaps are **named**, not numbered — see `../docs/paper-rewrite-audit.md` §3.2
- The containerised validation is **software-in-the-loop (SIL)**, never "the twin"
- Equation numbering from the original design draft is preserved where the equation survives,
  because the source code cites it
- Claims the paper must not make are listed in `../docs/paper-claims-inventory.md` §B
