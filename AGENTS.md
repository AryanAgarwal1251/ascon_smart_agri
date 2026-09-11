# AGENTS.md

This repository's agent instructions live in **[CLAUDE.md](CLAUDE.md)** — read it in full
before making changes. It applies to every AI coding agent, not only Claude.

The essentials (see CLAUDE.md for the rest):

- **[docs/design_paper.md](docs/design_paper.md) is authoritative.** Flag conflicts with it;
  never resolve them silently.
- **Follow the seven-phase gating discipline;** Phase 3 is a hard gate before federation. Stop
  and ask before starting a new phase's substantive logic.
- **Never hand-roll cryptography** — Ascon-AEAD128 from a KAT-verified library/reference only.
- **No remote CI** — gates run as local pre-commit hooks (ruff, vulture, mypy, pytest).
- **Never report accuracy alone; keys never in version control; dataset is CICIoT2023 only.**
- Do not commit or push unless the user asks.
