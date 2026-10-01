# Architecture diagrams (PlantUML)

Four diagrams, each as `.puml` source plus rendered `.png` and `.svg`:

| File | What it shows |
| --- | --- |
| `01_deployment` | What runs where, and the cryptography on each hop |
| `02_sealed_round` | One federated round, with the associated data on every frame |
| `03_verdict_routing` | The path a single sensor reading takes, and where it can end |
| `04_two_planes` | Training plane, runtime plane, and the feature-provenance boundary |

**The PNGs are already rendered and committed** — for Overleaf you need nothing else. Upload the
`paper/` folder and the images are there.

---

## How PlantUML works, and your three options

PlantUML turns a text description into a diagram. You write `.puml` source; something renders it
to an image. **Overleaf cannot render PlantUML itself** — it does not allow the shell access
PlantUML needs — so the image must be rendered first and uploaded, which is why the PNGs are
committed here.

### Option 1 — do nothing (recommended)

The `.png` files are already in this folder. `main.tex` includes them. Upload and compile.

### Option 2 — edit a diagram, no install

Run the link generator and open the URL in a browser:

```bash
./.venv/bin/python scripts/plantuml_links.py
```

It prints a `png` and an `svg` link per diagram. The link **contains the whole diagram**, deflate-
compressed into the URL, so plantuml.com renders it without anything being uploaded or stored.
Edit the `.puml`, re-run, open the new link, save the image over the old one.

To refresh every PNG and SVG in one go:

```bash
./.venv/bin/python scripts/plantuml_links.py \
  | awk '/^[0-9]/{n=$1} /png  /{print n, $2}' \
  | while read -r n u; do curl -s -o "paper/diagrams/${n%.puml}.png" "$u"; done
```

(`--fetch` does the same from Python, but fails behind an SSL-inspecting proxy; `curl` is the
reliable path.)

### Option 3 — render locally

Needs Java, which you have:

```bash
curl -L -o /tmp/plantuml.jar \
  https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar
java -jar /tmp/plantuml.jar -tpng paper/diagrams/*.puml
java -jar /tmp/plantuml.jar -tsvg paper/diagrams/*.puml
```

Nothing leaves your machine this way. Some diagram types want Graphviz as well; the four here do
not.

---

## Editing notes

- `scale 2` near the top renders at twice the default so the PNG is print-resolution at IEEE
  column width (347–711 dpi depending on the diagram).
- **A PlantUML directive cannot carry a trailing `'` comment on the same line.** `scale 2 ' note`
  silently produces an error image instead of a diagram. Put the comment on its own line.
- Colours match the paper's figures: blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`, ink
  `#52514e`.
- After editing, re-render **and look at the result** — a PlantUML syntax error renders as a small
  error image rather than failing loudly, and at a glance a 640×210 PNG looks like a diagram.

## Vector output

`.svg` is committed alongside each `.png`. The PlantUML server does not serve PDF (it answers
`302`), and EPS does not work with pdfLaTeX. If you want vector in the PDF, use the `svg` package
on Overleaf (`\usepackage{svg}`, `\includesvg{...}`) — otherwise the committed PNGs are print
resolution and need no conversion.
