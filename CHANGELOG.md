# Changelog

All notable changes to this repository are recorded here, newest first. This project follows
the seven-phase gating discipline of [docs/design_paper.md](docs/design_paper.md) (Section
III-J4) rather than semantic-versioned releases, so entries are grouped by date and tagged with
the phase they belong to. See [CLAUDE.md](CLAUDE.md) for the rule requiring this file to be
kept current.

## Results folder

[`results/`](results/) holds a paper-claim-vs-achieved writeup for each of the seven phases,
each figure traceable to the manifest in `artifacts/` it was pulled from rather than typed from
memory. Start at [`results/README.md`](results/README.md).

## Phase status

| # | Phase | Status | Exit criterion |
| - | --- | --- | --- |
| 1 | Characterisation report | Done, on both a subsampled/pre-split Kaggle mirror and the official UNB raw corpus (see 2026-09-13 entries) | Real columns/types, nulls, zero-variance, exact duplicate count, label vocab + counts, correlation matrix produced |
| 2 | Leakage-controlled preprocessing + four-stage feature selection | **Done** (Stage 4's knee sweep is wired but inert until Phase 3 supplies a detector — see the 2026-09-14 entry): subsample -> dedup -> split -> four-stage selection all run end-to-end on the real corpus; R3 gate passes on real data | `tests/test_leakage.py` green on real data ✅ |
| 3 | Centralised GRU + full evaluation | **DONE — gate CLOSED** (**hard gate**), verified by `scripts/check_phase3_gate.py` (exit 0). Baselines 1-3 of III-I1 reported over 3 seeds at W ∈ {1,16}; GRU macro-F1 **0.8297 ± 0.0013** at W=16 vs random forest 0.6855 and MLP 0.6070. Ablation answered: recurrence **earned its place** (+0.2322, 51× seed std) | Full evaluation protocol (macro-F1, per-class F1, balanced accuracy, MCC, confusion matrix, FPR; ≥3 seeds) reported **for baselines 1-3 of Section III-I1** ✅ |
| 4 | Three-client federated simulation, weighted FedAvg | **DONE — gate CLOSED**, verified by `scripts/check_phase4_gate.py` (exit 0). Two independent runs were merged; the reported figures are the **compute-matched** run (α=0.5, R=20, E=3, W=16, 60 local passes for every baseline): federated global macro-F1 **0.8308 ± 0.0150**, bracketed by local-only **0.7205 ± 0.0750** and centralised **0.8543 ± 0.0040**, recovering **82.4 %** of the gap; client-to-global gap positive for all three clients (+0.10 / +0.16 / +0.07). FPR **0.2975 ± 0.0512** — read it first. The earlier minimal-gate run is kept at `artifacts/manifest_phase4_minimal_gate.json`; its bracket is inverted because its baselines were not compute-matched. **Not done:** the α/E/aggregation ablation sweep of Section III-I3 | `test_fedavg_weighting.py`, `test_scaler_equivalence.py` green ✅; gate checker exit 0 ✅ |
| 5 | Telemetry simulation + feature-provenance adapter | **Done.** `telemetry/simulate.py` and `telemetry/provenance.py` implemented; G6 boundary verified on real data (200+ provenance refs checked, zero leaked into the training index) | Provenance adapter enforces G6 boundary ✅ |
| 6 | Ascon integration + alerting path | **Done.** `test_ascon_kat.py`, `test_ascon_tamper.py`, `test_nonce_collision.py`, `test_path_disjointness.py` all green -- the suite has **zero skips** for the first time in this project | `test_ascon_kat.py`, `test_ascon_tamper.py`, `test_nonce_collision.py`, `test_path_disjointness.py` green ✅ |
| 7 | End-to-end integration | **Done.** A federated global model was trained and saved for the first time in this project (macro-F1 0.8338, bit-for-bit identical to Phase 4's seed-0 result), and the full runtime pipeline ran for real: telemetry → held-out network features (G6) → streaming windows → the real model → Eq. (5) → routing → Ascon/alert. G1 held throughout (0 malicious-verdict messages reached the cloud) | Full pipeline run producing a manifest ✅ |
| 8 | Ascon-protected bidirectional weight exchange on physical hardware | **Steps 1–2 DONE; software twin DONE; 3–5 pending hardware.** Sealed weight channel + networked nodes (gated), and the whole topology as software: MQTT sensor contract, Pi runtime, TLS cloud receiver, ESP32/Wokwi sketch, Docker Compose. Headless end-to-end run: G1 held on the wire (cloud accepted == benign verdicts, 0 malicious). Plans: `docs/plans/phase8-hardware-federation.md`, `docs/plans/phase8-software-twin.md` | Two Pis + simulated third client complete R rounds with every weight frame Ascon-sealed both ways; tamper/replay frames rejected; hardware global model == in-process FedAvg bit-for-bit; local GRU routes on its own verdict with 0 malicious payloads on the cloud path |
| 9 | Multi-dataset generalisation (CICIoT2023 + CICIoMT2024; Edge-IIoTset registered, not trained on) | **Third design COMPLETE (2026-09-23).** Mixed farms (every farm draws from every corpus) raised in-distribution CICIoT2023 **0.7742 → 0.8211** and CICIoMT2024 present-only **0.8034 → 0.8714**; adding the second corpus now costs 0.020 against the single-corpus ceiling rather than 0.066. Leave-one-dataset-out unchanged and still failing (0.0853 / 0.0763) — reported as a negative result. Manifest `artifacts/manifest_phase9_generalised_percorpus_mixed.json`. Prior: **second design COMPLETE (2026-09-22), mixed result.** Per-corpus scaling fixed the in-distribution collapse (CICIoT2023 0.603 → **0.7742 ± 0.0119** federated, pooled twin **0.8390 ± 0.0035** = Phase 3's single-corpus number) and the driver reproduces Phase 4 (0.8407 ± 0.0150 vs 0.8308). **Leave-one-dataset-out failed: 0.0858 ± 0.0347 on held-out CICIoT2023 (FPR 0.95), 0.1380 ± 0.0485 present-only on held-out CICIoMT2024 (FPR 0.82)** — no transfer to an unseen testbed, reported as a negative result. Manifest `artifacts/manifest_phase9_generalised_percorpus.json`. Prior status: **second design running (2026-09-22).** The first full run (2026-09-21, all three corpora, global scaler, 13 features) gave in-distribution 0.60 / 0.56 / 0.12 and leave-one-dataset-out ≈ 0 -- per-corpus feature scales, and an Edge-IIoTset whose reconstructed windows are near-constant. Decided: Edge-IIoTset `not_for_training`, all 16 features back, per-corpus standardisation, K = 3 across corpora (Dirichlet within one), a pooled-centralised twin on the same windows, LODO over the two corpora, at Phase 4's budget. Plan §5: `docs/plans/phase9-multi-dataset.md` | Each corpus characterised and reconciled with the registry; per-corpus R3 gate green; leave-one-dataset-out results over ≥ 3 seeds |

Most modules under `src/ascon_smart_agri/` are still typed stubs: they `del` their unused
parameters and raise `NotImplementedError("Phase N: ... not implemented yet.")`. The exceptions
are the Ascon-AEAD128 crypto core (`crypto/ascon_aead.py`), implemented ahead of its phase gate
(see the 2026-09-12 Phase 6 entry below), `data/characterize.py` (Phase 1), and
`data/subsample.py`/`data/dedup.py`/`data/split.py`/`features/selection.py` (Phase 2, complete)
-- see the 2026-09-14 and 2026-09-13 entries below.

## 2026-10-01

### Citation audit finished: 32 of 38 verified, and the rest are deliberate

Six more entries resolved at source, and the remaining notes were classified rather than left as
an undifferentiated backlog --- because most of them are not work owed.

**Resolved this round.** Okey et al. is ISWCS 2024, pp. 1--6, with DOI; the landing page that
first surfaced it did not name the venue, the IEEE record does. LWE-IoT is Suleiman, Javeed and
Raza, *Int. J. Circuit Theory and Applications* 2026, and its edge nodes are ESP32 with the
broker on a desktop --- the same shape of measurement as ours. Khraisat et al. has six authors in
*Discover Internet of Things* 5:72. The granted US patent is 12,301,597, *Network edge digital
twin for IoT attack detection*, inventors Alhazmi and Alakeel, assigned to King Saud University;
it uses our corpus with a random forest at the edge, which is what makes it the closest granted
patent by corpus and placement. Two stale page-range markers were cleared.

**What remains, and why.** Three entries still want a number we can look up: two Indian patent
applications and nothing else. **Four are standing cautions that should stay**, because they are
doing their job: they forbid citing a figure we have not read in the source --- the survey's "up
to 76 % F1 drop", Farm-Flow's 101-feature count, the IIoT paper's unnamed datasets, and a volume
number the journal has not yet assigned. **A check confirms none of those figures appears
anywhere in `main.tex`.** The notes prevented exactly what they were written to prevent.

**Two entries remain incomplete and should be cut**, as previously recommended: the
distributed-IoT differential-privacy paper and the multi-level federated Industry 4.0 patent.
Each is cited once inside a list, neither has resolvable authorship or venue, and dropping both
leaves 36 references.

### The Ascon section rewritten against the published standard, which also caught our own citation error

The user supplied the actual NIST SP 800-232 document. Checking the paper against it resolved the
last outstanding citation marker, corrected an error of ours, and produced genuinely new material.

**An error of our own making.** Our bibliography entry for SP 800-232 listed four authors and
carried a note claiming it had been checked at source. The published document's own "How to cite"
block gives **five**: Meltem Sonmez Turan, Kerry A. McKay, Donghoon Chang, Jinkeon Kang and John
Kelsey. **Kerry A. McKay, the second author, was missing.** This is the fourth citation-accuracy
problem the audit has surfaced and the first that is ours rather than the review paper's.

**Claims now confirmed from the primary source**, and the `\verifycite` marker on the CAESAR
sentence removed: Ascon-128 and Ascon-128a were selected in 2019 as the *first choice* for the
lightweight authenticated-encryption use case in the final CAESAR portfolio. The variant-mismatch
warning is confirmed and sharpened --- Ascon-AEAD128 **is based on Ascon-128a**, and of the seven
changes the standard enumerates, two break interoperability silently: reformatted initial values
and a switch **from big-endian to little-endian**. The section now also states why a round-trip
test does not catch this (it encrypts and decrypts with the same wrong constants), and cites the
Cryptographic Algorithm Validation Program that SP 800-232 Sec. 6 itself names as the provenance
of the known-answer vectors our gate asserts.

**New material the standard made available (Section II-D, "What the standard requires of a design
that uses many keys").** Two conformance requirements bear on an architecture that issues a key
per client *and per direction*:

- **Multi-key security is quantified, and the cost of federating is small.** The standard gives
  $(128 - \log_2 u)$-bit security for $u$ independent keys. Our design has $u = 2K$, so the price
  is $\log_2 2K$ bits: 125.4 bits at $K = 3$ and 121.4 bits at $K = 50$. **Scaling the federation
  sixteenfold costs four bits.** The nonce-masking option would restore the full 128 bits; we
  decline it, because it needs a second key per channel and forfeits context-commitment security,
  and the equation shows there is nothing worth buying back.
- **The key-lifetime limit is unreachable here.** The standard caps one key at $2^{54}$ bytes and
  requires rekeying there. At our measured 135,942 bytes per client per direction per round that
  arrives after $1.3 \times 10^{11}$ rounds --- about $6.6 \times 10^{9}$ runs of the twenty-round
  schedule. The companion $2^{96}$ decryption-failure bound is likewise far above our zero
  rejections; the aggregator counts them because a nonzero count is a signal, not because the
  bound binds.

The permutation description is now accurate to the specification as well: five 64-bit words, 12
rounds at initialisation and finalisation against 8 while absorbing, rate 128 against capacity
192, and forward-only use so no inverse permutation need exist.

### The abstract rewritten to define the project rather than list its results

The previous abstract opened on federated learning as a technique and reached the agricultural
setting only by implication, so a reader met the method before the problem. It now opens where the
project does: a smart farm is a cyber-physical system, a soil-moisture reading decides whether a
pump starts, and a corrupted value wastes water rather than merely misleading a chart. From there
it states the two requirements that arrive together, why federation answers the first, the
exposure federation then introduces --- the parameters crossing the same untrusted network the
detector protects --- and the unexamined question of what the verdict should do to the data path.

Only then does it describe the system, and it now names what the associated data binds and why
(a frame cannot be replayed, reflected or reattributed) rather than asserting that the channel is
sealed. The results keep the per-client framing, since the figure an operator actually needs is
whether *their* farm gains, and the client-count sweep is stated as the aggregate concealing the
rare-class collapse rather than as two separate numbers. Both negative results and the
provenance boundary stay in the abstract, where a reader meets them before the claims rather than
after.

It went through three lengths before settling: 564 words, then 434, now **335**. That is still
above a strict 250-word venue limit, and if a conference is chosen it will need another pass ---
but the detail it would lose is all restated in the introduction, so nothing is at risk.

### A hardware deployment section, and the citation audit begins

**`main.tex` gains Section VII-C, "Hardware deployment"** (1,430 lines, 12 figures, 8 tables).
It states that the containerised topology maps one-to-one onto physical hardware and that moving
between them changes where processes run and nothing else; carries the bill of materials as a
table; and embeds the new pin-and-wiring figure. Two constraints are called out because neither
announces itself at run time: the moisture probe must be capacitive, since resistive probes
corrode within weeks in wet soil, and it must be read on an ADC1 pin, because the ESP32's second
ADC is disabled while the Wi-Fi radio is active and a permanently associated node would read
noise and report it as a moisture value.

The section also names **the three quantities only hardware can supply** -- per-round wall-clock
time on a gateway CPU, energy per round and per inference, and the behaviour of the sensor hop
over real Wi-Fi -- so the paper says what it cannot measure rather than leaving the reader to
assume.

**Citation audit: 17 of 37 entries now checked at source, 19 outstanding, 2 incomplete.** Six
were resolved this round, and two of them corrected errors:

- **Ruzafa-Alcazar et al.** --- an earlier note in this bibliography suspected the review paper's
  year was wrong and that IEEE TII 19(2) was 2023. It is **2021**, exactly as the review paper had
  it. The note was wrong, not the citation.
- **Bai et al.** --- the first author is **Ye** Bai, not "Yu" as a previous guess here recorded,
  and there are six authors rather than one. Their experiments use UNSW-NB15 and report the random
  forest achieving the highest classification accuracy, which is precisely the comparison our own
  baseline table puts to the test, so the entry now records that.
- **Karunamurthy et al.** --- the review paper's "et al." concealed a fourth author.
- **Belarbi et al.** --- GLOBECOM 2023, Kuala Lumpur, pp. 237--242, with DOI.
- **Bilal et al.** --- the closest published work now has its six authors and DOI; it is dated
  January 2026, not 2025.
- **Ferreira et al.** --- Farm-Flow is *Computers and Electrical Engineering* vol. 121 art.
  109892 (2025); the key was renamed and the citation in `main.tex` updated with it.

### A pin and wiring figure for the hardware node

`scripts/make_figures.py` gains `fig_pinout`, drawn rather than photographed so the pin
assignments stay legible at IEEE column width and match
`firmware/esp32_sensor/diagram.json` exactly, wire colours included. It shows all six wires, the
two that carry signal (DHT22 data to GPIO 15, soil moisture to GPIO 34), and the constraint that
is easiest to get wrong: **GPIO 34 is on ADC1, and ADC2 stops working entirely while Wi-Fi is
active**, so a node that is always associated would silently read noise from an ADC2 pin.

Two bugs were worth the detour. The figure first rendered as 7 of 7 rather than 8 of 8, because
the registration line sat below the `if __name__` guard and so executed after `main()` had already
iterated the registry. And the first clean render put pin labels on top of the box titles and ran
a wire straight through one of them; the fix moves the peripheral boxes right to open a gutter,
right-aligns the box titles, and gives each pin label a surface-coloured backing box so no wire
can cross text. Rendering and looking at the output caught both --- neither is visible from the
source.

### Tables were overflowing the page; fixed, and the references audited for relevance

The user compiled the draft and found two tables running past the right margin with their last
column cut off. The cause was column specification: `l` columns size themselves to their content,
so a long cell pushes the table past the margin with no warning -- LaTeX reports an overfull box
and carries on. The reviewed-corpus table was `@{}llllp{6.3cm}@{}` in a full-width float and the
assumptions table `@{}llp{3.1cm}@{}` in a single column; both had four or five centimetres of
unbudgeted `l`-column width.

**Every column in every table is now a fixed-width `p{}`**, with `\tabcolsep` reduced so the
budget goes to content rather than padding, and a check added that sums each table's declared
widths plus padding against its container (8.89 cm for a column, 18.19 cm full width). All seven
tables now fit, the tightest at 18.14 of 18.19 cm.

**Reference audit.** All 37 entries are cited, and the citation context of each was inspected
rather than assumed. They fall into three tiers: fifteen are load-bearing, in that a claim or a
design decision rests on them; eighteen are supporting context; and **two are weak enough to
consider cutting** -- a multi-level federated Industry 4.0 patent filing and a distributed-IoT
differential-privacy paper, each cited once inside a list and each doing little work. Those two
are also, as it happens, the two still marked `**INCOMPLETE**` in the bibliography for want of
authors or a resolvable venue. Dropping them would resolve both problems at once and leave 35
references, still well clear of the target.

### Paper batches 2-5: the draft is complete

`main.tex` goes 816 -> **1,368 lines** and the last `\todo` is gone. Every section is written
from a committed manifest.

**Batch 2 -- equations, 11 -> 33.** The formal problem statement (window matrix, binary
projection, federated objective), the subsampling cap and imbalance ratio, the four GRU gate
equations and the layer-normalised head, mutual information and reciprocal rank fusion for feature
selection, and the full AEAD construction: encryption, decryption returning the bottom element,
both associated-data layouts, the nonce birthday bound, and wire overhead. Macro-F1 and FPR are
now defined rather than assumed.

The one that mattered most is **Eq. (8), the leakage-inflation bound** `Delta ~ delta*rho*(1-Acc)`.
It is the entire argument for deduplicating before splitting, and the paper now makes the point
the formula carries: the inflation scales with `1 - Acc_true`, so **a detector is flattered most
precisely on the classes where its performance matters**.

**Batch 3 -- threat model and three tables.** Constraints C1-C4, the three channels with their
distinct adversaries, and the scope lists. Table~I ties each attack family to its consequence in a
farm, which is the only place the paper connects a class label to a physical outcome. Table~II
separates assumptions that are properties of the world from artefacts of working without hardware.
Table~VI carries the risk register with its controls.

**Batch 4 -- results.** Recurrence is worth +0.2322 macro-F1 against an otherwise-matched
perceptron, fifty-one times the seed standard deviation. The federated bracket, 82.4 % of the
local-to-centralised gap recovered, with FPR read first. Per-client gains, all positive, and the
seed where two starved clients went 0.6145 -> 0.8145. The communication argument, now stated as
specific to small K because it reaches 13.59 MB per round at K = 50. The dataset section reports
both rejections as measurements, including the 7.33x extractor window-size difference and the
three features whose ranges are disjoint. Cross-corpus transfer is framed as a replication, with
the explicit note that **our semantics defect is not its cause** -- standardised schemas do not
rescue transfer for anyone.

**Batch 5 -- introduction, limitations, conclusion.** The introduction frames from the deployment
and names the parameter exchange as the channel this work introduces. The limitations section
states nine of them in our own voice before a reviewer can, including that leave-one-device-out
could not be run because the corpus carries no device attribution, and that gradient inversion and
poisoning are live against this design.

Final state: 1,368 lines, 9 sections, **33 equations, 7 tables, 4 algorithms, 11 figures, 37
references all cited**, zero dangling cross-references, all environments balanced, two
`\verifycite` markers left on claims still unconfirmed.

### Paper batch 1: the literature review, written

`main.tex` grows from 592 to 816 lines. Section II now carries the review protocol with its
selection-parameter table, the full twenty-study corpus as Table~V, five literature threads, and
the patent landscape. **35 of the 37 bibliography entries are now cited**; the two that are not
belong to the dataset-selection results still to be written.

Threads A--D follow the submitted review paper, with its limitations restated as the named gaps
rather than as `L1`--`L5`. **Thread E is new** and exists because our own results demanded it: it
sets out why cross-corpus transfer fails for everyone, that four corpora sharing a *single*
standardised 43-feature schema still lose 56.28 % on average, that the failure is strongly
asymmetric (94.83 % one direction, 4.90 % reversed), and that the published remedy is combined
multi-corpus federated training recovering about 90 % --- the design this work reached
independently. It also records why we keep weighted FedAvg and a GRU despite FedNova's 15--25 %
communication saving and a Transformer's 1--2 point gain, and introduces leave-one-device-out as
the generalisation question a federated farm actually faces.

**Two citation corrections are now in the prose, not just the `.bib`.** Thread C states plainly
that Oztuerk et al. build a *chaos-based Ascon variant* and that their result is evidence
Ascon-class cryptography runs on gateway hardware, **not** evidence about standardised
Ascon-AEAD128 over MQTT --- the claim the review paper made. Table~V's S19 row says the same in
its relevance column. El-Hajj and Gebremariam are cited as the published journal article with
their actual memory figures (about 1 KB additional RAM and Flash for Ascon against roughly 8 KB
RAM and 5.6 KB Flash for AES-GCM) rather than as an unnamed preprint.

Checks after writing: no dangling cross-references, all environments balanced, braces matched,
every `\cite` key resolving.

### The submitted review paper supplies the authors and 25 references; verification finds two wrong citations

The user provided the submitted review-and-design paper. Its author block is now in `main.tex`
(Abhishek Jadli, Aryan Agarwal, Advait Amit Malviya, Vijayprabhakaran K; VIT Chennai), and its
reference list [1]-[25] is merged into `paper/references.bib` alongside the twelve entries added
from the generalisation literature review. **37 entries**, comfortably past the 25 the user asked
for.

**Twelve of the inherited entries were incomplete** -- missing authors, volume, or a resolvable
venue -- which is not a submittable state. Verification so far has closed five of them and
surfaced **two citations that do not say what the review paper claims**:

- **Ref [19]** is cited as confirming "the standardised variant is usable over MQTT". It does not.
  Oztuerk et al., *Applied Sciences* 15(19) art. 10641, build a **chaos-based Ascon variant**,
  deriving keys and nonces from a Zaslavsky chaotic map and claiming higher security than standard
  Ascon, tested on Raspberry Pi 3B with ROS 2 robots. It is good evidence that Ascon runs on
  Pi-class hardware -- which we can use -- but it is not evidence about standardised
  Ascon-AEAD128 over MQTT. The sentence must be rewritten.
- **Ref [18]** is cited as a 2024 preprint. It is a **published journal article**: El-Hajj and
  Gebremariam, *Network* (MDPI) 4(3):260-294, 2024. Its actual numbers are more useful than the
  review paper's summary -- Ascon has lower execution time than AES-GCM and needs about 1 KB more
  RAM and Flash where AES-GCM needs 8 KB more RAM and 5.6 KB more Flash, at equivalent security.

Also completed: **[10]**, the paper the review calls closest prior work, is Okey, Dadkhah,
Rodriguez and Kleinschmidt -- a federated ensemble over CNN, GRU and LSTM with wFedAvg and
wFedProx, reaching 98.25 % accuracy on CICIoT2023 and 93.36 / 95.66 % with LSTM-GRU on
CICIoT2023 / FLNET2023; **[11]** is Khraisat, Alazab and Alazab, *Discover Internet of Things* 5
art. 72; **[20]** is Fathy and Ali, *Sensors* 23(4) art. 2091.

Every entry now carries a `% CLAIM:` line stating what we attribute to it and a `% VERIFY:` line
naming what is still unchecked, so no citation can reach submission on trust alone. Eleven are
checked at source; two remain `**INCOMPLETE**` (a 2025 IEEE item with no authors or venue, and an
Industry 4.0 patent filing with no jurisdiction or number).

### Figures and architecture diagrams, generated rather than drawn

Seven plots and four PlantUML diagrams, all committed, and `main.tex` now includes all eleven --
no placeholder boxes remain.

**`scripts/make_figures.py`** reads every figure straight from a committed manifest, so running it
is the check that the paper's prose and its plots agree. Charts follow the project's
visualisation guidance, and one planned figure had to be redesigned because of it: the client-count
figure was specified as macro-F1 and rarest-class F1 on **twin axes**, which is the most common
charting error. Both series are F1 on $[0,1]$, so they now share one axis -- and that is the point
of the figure, since the aggregate concealing the rare-class collapse is only legible on a common
scale. Traffic-per-round, which genuinely does not share that scale, became its own figure.

The three-colour palette was **run through the validator rather than eyeballed**: all checks pass
on the all-pairs colour-vision and normal-vision floors, with one documented warning (the aqua slot
sits below 3:1 contrast on the light surface), relieved by direct labels and by the data appearing
in Table I. Every series also carries a distinct marker and line style, because an IEEE paper is
frequently read in greyscale and colour must never carry identity alone.

Rendering and looking at the output caught a defect the validator cannot see: in the first
client-count plot the "pigeonhole bound 42" annotation collided with the weakest-class endpoint
label in the lower-right corner. Moved and re-rendered.

**`paper/diagrams/`** holds four PlantUML diagrams -- deployment, one sealed federated round, the
runtime verdict path, and the two planes with the feature-provenance boundary -- as `.puml` source
plus rendered `.png` and `.svg`. **`scripts/plantuml_links.py`** compresses a diagram's source into
a plantuml.com URL (raw DEFLATE plus PlantUML's own base64 variant, whose alphabet differs from
standard base64 -- an ordinary `b64encode` yields a link that silently fails), so a diagram can be
edited and re-rendered with nothing installed. Overleaf cannot run PlantUML, which is why the PNGs
are committed.

Two traps are recorded in `paper/diagrams/README.md` because both cost time: **a PlantUML
directive cannot carry a trailing `'` comment on the same line** -- `scale 2 ' note` produces a
small error image rather than failing loudly, and at a glance a 640x210 PNG looks like a diagram --
and the PlantUML server does not serve PDF, answering `302`, while EPS is useless to pdfLaTeX. The
committed PNGs render at 347--711 dpi at IEEE column widths, so no conversion is needed.

`matplotlib` is added to the `dev` extra with a note that nothing under `src/` imports it, so a
Raspberry Pi runtime install does not pull it in.

### An Overleaf-ready IEEE paper skeleton, with the measured sections already written

`paper/main.tex` (new, 508 lines), `paper/references.bib`, `paper/figures/`, `paper/README.md`.
Zip `paper/` and upload to Overleaf; pdfLaTeX plus BibTeX, and `IEEEtran` is preinstalled there.

It is a working draft rather than an outline: everything that rests on a committed manifest is
written, and everything still owed is marked. Written now --- the abstract with the real figures,
the contribution list, the **Ascon section** the user asked for (sponge-based duplex construction
from the designers' Journal of Cryptology paper, NIST SP 800-232 standardising the Ascon-AEAD128
variant this system actually uses, and the trap that the standardised variant differs from
pre-standard Ascon-128/128a in its IVs and endianness so a popular PyPI package is not
interoperable), the eight **named** research gaps, nine equations, two algorithm floats, the
client-count results table and its three-regime reading, and the whole software-in-the-loop
section including the vacuous-pass analysis. Twenty-one `\todo` markers carry the rest, mostly
narrative sections that depend on the citation audit.

Three things were built to keep the draft honest:

- a `\figbox` placeholder so a figure that does not exist yet renders as a labelled box and the
  document always compiles --- no broken `\includegraphics` and no silent omission;
- a `\verifycite` marker for any citation whose record or attributed claim is still unconfirmed,
  so an unverified reference is visible in the PDF rather than buried in the `.bib`;
- an automated check, run before committing, that every `\ref`/`\eqref` resolves (no dangling
  references) and that every `\cite` key exists in the bibliography. Both pass.

Two equations the audit said were owed are now written: the **weight-channel associated-data
layout** (Eq. 'weightad'), which is the paper's central security construction and had no equation
in the design draft, and the **pigeonhole bound** on client count from the rarest class's block
count, which gives 42 and is exactly where the sweep's second descent begins. The bytes-per-round
equation is stated with its measured discrepancy (811,200 predicted against 815,136 measured, the
0.5 % being safetensors framing) rather than as a clean prediction.

The naming decisions from the audit are applied throughout: gaps are named, not numbered, and the
containerised validation is **software-in-the-loop**, defined precisely on first use --- production
node software, unmodified, as separate OS processes on a real network at the target CPU
architecture --- with "container timings are not gateway timings" in the same paragraph and
hardware-in-the-loop named as future work.

### The client-count sweep completed: 21 points, and the prediction was right after all

`artifacts/k_sweep_results.json` and `manifest_ksweep_default.json`. K in {3,5,10,20,30,40,50}
x 3 seeds at Phase 4's budget on the CICIoT2023 cache, 21.2 hours.

| K | macro-F1 | FPR | rarest-class F1 | seq/client | MB/round | empty clients |
| - | --- | --- | --- | --- | --- | --- |
| **3** | **0.8367 +- 0.0107** | 0.2988 | **0.642** | 399,424 | 0.82 | 0 |
| 5 | 0.7835 +- 0.0340 | 0.3366 | 0.564 | 238,361 | 1.36 | 0 |
| 10 | 0.7974 +- 0.0195 | 0.2144 | 0.557 | 118,012 | 2.72 | 0 |
| 20 | 0.7866 +- 0.0260 | 0.1765 | 0.531 | 58,618 | 5.43 | 0 |
| 30 | 0.7863 +- 0.0001 | 0.3149 | 0.513 | 38,979 | 8.15 | 0 |
| 40 | 0.7778 +- 0.0091 | 0.3184 | 0.466 | 29,215 | 10.87 | 0 |
| **50** | **0.7577 +- 0.0134** | **0.3882** | **0.386** | 23,355 | 13.59 | 0 |

**An earlier reading in this changelog called the curve a plateau that contradicted the analytical
prediction. The final two K values correct that.** The curve has three phases: one step down from
K = 3 to K = 5 as per-client data falls below the 400k sequence cap (399,424 -> 238,361); a genuine
plateau from K = 5 to K = 30, where 6x more clients costs 0.004 and K = 30's seed spread is
+- 0.0001; and then a **second descent from K = 40 to K = 50** (0.7863 -> 0.7778 -> 0.7577). That
descent begins immediately after the pigeonhole ceiling of 42 and is steepest at K = 50, which is
where the analytical knee of K* ~ 48 sits. The prediction was not wrong; reading the curve at
K <= 40 was premature.

**The rarest class declines monotonically at every step and never recovers:** 0.642, 0.564, 0.557,
0.531, 0.513, 0.466, 0.386 -- **-0.256 absolute, a 40 % relative fall**, while macro-F1 moves only
-0.079 because the common classes (Mirai at 0.998 throughout) hold the average up. **Reporting
macro-F1 alone would have hidden a 40 % collapse in the hardest class**, which is an
evaluation-methodology finding in its own right and belongs in the paper beside the
vacuous-verification one.

**Zero empty clients at every K, including 50**, so this is dilution rather than literal
starvation: the rare class is never absent from a client, it is ground down. FPR is worst at K = 50
(0.3882) and tracks the rare-class collapse more honestly than macro-F1 does. Communication scales
linearly from 0.82 to **13.59 MB per round** (16.6x), and since accuracy is already 0.079 down at
K = 50, the federation benefit degrades before the predicted communication break-even at K ~ 51 is
even reached.

**For the hardware configuration this is a clean result: K = 3 is the sweep's best point**, 0.8367
against the centralised ceiling of 0.8543 -- 0.018 below it -- with the best rare-class F1 of any
client count. The two-Pi budget constraint turns out to be the accuracy-optimal choice.

### Paper rewrite begins: audit of the old draft, and a bibliography built from scratch

The user asked to write the paper properly from the beginning -- reviewing the existing draft and
correcting what is wrong, writing the equations, giving every measurement a graph, adding an Ascon
section from the designers' own paper, writing algorithms, renaming the internal shorthand, and
checking that every citation opens and actually says what we attribute to it.

**`docs/paper-rewrite-audit.md`** (new) is the worklist. The draft in `design_paper.md` was written
before the system existed and says so: "we claim no new algorithm; the contribution is the
composition and the rigour of its specification." Seven of its statements are now contradicted by
our own measurements -- most importantly **it describes Ascon as protecting the telemetry, when the
architecture protects the model weights in both directions and leaves the cloud hop on plain TLS**.
Its single global scaler, its implicit single-corpus assumption, and its "K = 3 by assumption" all
need amending. One passage deserves promotion rather than rewriting: its insistence that the
malicious handler hold no reference to the cloud client, with an automated test asserting no
encrypted payload on a malicious-only run, is exactly what the implementation does and is why the
property survived a year of refactoring.

**Two renamings, both as the user asked.**

*"Twin test" becomes software-in-the-loop (SIL) validation*, which is the established term for the
real software running against simulated inputs before the target hardware exists, and it sits in a
recognised SIL -> HIL progression with the pending Raspberry Pi run. The paper will define it
precisely on first use so no reader assumes more than was done.

*`G1`-`G6` become named gaps, written out in full* where introduced and referred to by name
thereafter: the verdict-routing gap, the unbounded-privacy-claim gap, the partition-specification
gap, the missing-lower-bound gap, the sequence-construction gap, and the feature-provenance gap.
Two further gaps are ours, both found by measurement: **the vacuous-verification gap** (safety
properties reported satisfied without reporting how many opportunities to violate them occurred)
and **the cross-corpus semantics gap** (corpora combined on column *names*, so a shared schema can
hide quantities differing by a factor of 2.4e11).

The audit also fixes the equation inventory -- existing numbering is preserved because the code
cites it, and three equations are owed: the weight-channel AD layout, per-packet normalisation, and
the pigeonhole bound on client count -- and lists 13 figures and 5 algorithm floats against the
manifests that supply them.

**`paper/references.bib`** (new) is the reference list, rebuilt from zero because **every inline
citation in the old draft is a broken `[URL](#page-0)` conversion artefact and none can be trusted**.
Sixteen entries, each carrying the claim we attribute to it and whether that claim was read in the
source or still needs checking; fourteen carry an outstanding VERIFY note, which is the honest
state. The two Ascon entries are fully checked: the designers' Journal of Cryptology 34(3) article
33 (2021), and **NIST SP 800-232** (Turan, Chang, Kang, Kelsey; final 13 August 2025), which
standardises **Ascon-AEAD128 -- the exact variant this system uses**. Our known-answer vectors come
from the official `ascon-c` reference implementation, and a detail worth a sentence in the paper:
SP 800-232's Ascon-AEAD128 differs from the pre-standard Ascon-128/128a in its IVs and endianness,
and the PyPI `ascon` package implements only the old variants -- which is why the backend is a
vendored reference implementation gated on the KAT.

### Literature review: our cross-corpus failure is the published result, and the headline changes

The user asked for proper research before any further decision. `docs/generalisation-literature-review.md`
(new) is that research, and it overturns how the previous two entries framed the problem.

**Zero-shot cross-dataset transfer fails for everyone.** The decisive study (arXiv 2205.04112)
converts four NIDS corpora to **one standardised 43-feature NetFlow schema** -- built precisely to
remove feature-schema mismatch as an excuse -- and still measures **56.28 % average performance
decay**, sometimes near 0 % F1, with Extra Trees scoring 94.83 % in one direction and 4.90 %
reversed. SHAP attributes within-dataset performance to shortcut features that do not transfer.
Corroborated for IIoT (arXiv 2607.00553) and in an 89-dataset survey reporting F1 drops up to 76 %
(arXiv 2502.06688).

**So the feature-semantics bug is still a bug, but it is not why LODO fails, and fixing it will not
rescue LODO.** The previous entry struck the LODO number as contaminated; it is now reinstated and
reframed as a **replication of a known result, with citations** -- which is stronger than either
hiding it or presenting it as novel.

**What does work is what we already built.** The closest published work (Nature Scientific Reports,
federated IDS over Edge-IIoTset + CIC-IoT2023 + TII-SSRC-23) reports in-domain 96.5/95.2/93.0 %
macro-F1, **up to 30 pp loss** cross-dataset, and **~90 % macro-F1 across all corpora under
combined multi-dataset federated training**, recommending exactly that. This is our mixed-farm
design, and it independently reproduces our own 0.7742 -> 0.8211 result. Two of their secondary
findings are now recorded as alternatives the paper should acknowledge: FedNova cuts communication
15-25 % against FedAvg, and a Transformer beats an LSTM by 1-2 points.

**A better generalisation test exists, and we cannot run it.** The paper this project already cites
for the Edge-IIoTset placeholder artefact goes considerably further than we knew: it rebuilds the
corpus from raw captures as **AgriEdge** (1,276,122 rows, five devices with full attribution, no
column separating classes above 0.0288, MIT licence on GitHub and Zenodo) and runs
**leave-one-device-out**, locating the generalisation boundary at the perception/actuation layer
where random forest falls 0.9988 -> 0.5083. Leave-one-device-out is the right question for a
federated farm deployment -- a new farm is unseen devices, not an unseen testbed -- and it needs
device attribution rather than four corpora. **CICIoT2023's published CSVs cannot support it**, which
was verified directly: 47 source files across 34 attack-type directories, no device identity in
either the feature columns or `source_file`.

**Agriculture-specific corpora that do exist:** **Farm-Flow** (real AG-IoT testbed, >1 M flows, 101
features, 8 attacks including **MQTT Flood** -- an attack on the protocol our ESP32s actually speak,
absent from both CIC corpora, Zenodo) and **AgriEdge**. Neither shares CICIoT2023's 39-column DPKT
schema, and no NetFlow conversion of CICIoT2023 exists -- but per the first finding, schema
standardisation does not buy generalisation anyway, so compatibility matters only for pooling into
one model.

Recommendation recorded: do not chase zero-shot LODO; fix the semantics bug on its own merits (the
7.33x window-size difference between the two CIC extractors is itself reportable, and per-packet
normalisation recovers four of the eight broken columns); make combined multi-dataset FL the
headline; and shift the generalisation claim to leave-one-device-out only if Farm-Flow or AgriEdge
is added. Do not add TII-SSRC-23 or the NF-v2 family -- not agricultural, and they would not deliver
zero-shot transfer either.

### Three of the sixteen features do not mean the same thing in the two corpora

The user asked how to make the model generalise across IoT testbeds. Diagnosing before proposing
turned up a defect that invalidates several numbers reported over the past two days, so the
measurement comes first.

Read straight out of `artifacts/phase9_cache.npz`, before any scaling:

| Feature | Phase 2 rank | CICIoT2023 | CICIoMT2024 |
| --- | --- | --- | --- |
| `Variance` | 5 | 0 … 1.13e8 | **0 … 1** (100 % of rows <= 1) |
| `Header_Length` | 2 | 0 … 60 (median 20, a correct IP header) | **0 … 9.89e6** |
| `IAT` | 6 | 2.4e-7 … 4.7e4 | **4.7e-6 … 1.69e8**, with 79.6 % of rows within +-1 % of 8.47e7 |

`IAT` medians differ by a factor of 2.4e11. These are not distribution shifts; they are different
quantities wearing the same column name. **Three of Phase 2's top six features are affected.**

This was a recorded risk that had not been discharged: `data/datasets.py` states that every alias
for the new corpora "was written from the published documentation, not from the files", and the
2026-09-22 entry lists re-verifying CICIoMT2024's feature semantics as an open follow-up. Only the
`Duration -> Time_To_Live` alias was ever checked against real values. Per-corpus standardisation
then hid the problem in-distribution, because a model trained and tested inside one corpus never
sees the other's scale -- which is why it survived until a cross-corpus evaluation.

**Withdrawn as contaminated**, and struck in place rather than deleted: the leave-one-dataset-out
figures (0.0853 / 0.0763), the 0.023 and 0.070 costs attributed to adding CICIoMT2024, and the
two-corpus pooled twin. We came close to publishing a wrong negative result -- the claims
inventory had listed the LODO failure as a contribution.

**Unaffected:** everything that never crosses corpora. Phase 3's 0.8297 and the recurrence
ablation, Phase 4's 0.8308 and the 82.4 % gap recovery, the entire K sweep, the per-client
federation gain on CICIoT2023, Phases 7 and 8, G1, the Ascon work, and Edge-IIoTset's 0.9822 on
its own native schema. CICIoMT2024's label coverage (6 of 8 families) is derived from filenames,
not feature columns, so it stands.

**The dataset decision survives on two of its three legs.** Domain coherence (a medical testbed
against an agriculture deployment) and the absence of any attack family CICIoT2023 lacks both
hold; the measured-degradation leg is withdrawn. `docs/dataset-selection.md` is corrected in place.

`docs/generalisation-plan.md` (new) then sets out the actual route, with the fix as step 0 because
domain adaptation layered on mismatched features would only be correcting a data defect: verify
all 39 aliases against real values, add a cross-corpus range-overlap test that fails loudly on a
10^11 disjointness, re-run LODO for the first honest generalisation number, then rank/quantile
features, shape features instead of magnitudes, and domain-adversarial training.

It also names a real tension. **With two corpora, leave-one-dataset-out trains on exactly one
domain** -- that is domain *transfer*, the hardest possible setting, not domain *generalisation*,
which needs variation across several source domains to learn invariance from. So if cross-testbed
generalisation is to be a headline claim, more corpora are required -- not as extra attack
coverage, which they do not provide, but as **additional domains**, which is precisely what is
missing. That is a different justification from the one they were rejected on, and the choice
between the two headline claims is the user's.

Also corrected: the claims inventory had listed "validated on Raspberry Pi hardware" among claims
the paper must never make. The hardware demo is planned, so it now sits in a new A10 "pending"
section alongside the measurement that will fill it in, and only *reporting container timings as
Pi timings* remains forbidden.

### Paper groundwork: a claims inventory built before any drafting

The user asked to start the IEEE paper once the K sweep lands. `docs/paper-claims-inventory.md`
(new) is the first artefact, written deliberately *before* prose: every number the paper may use,
traced to a committed manifest, and -- the half that matters more -- every claim the paper must
**not** make, with the wording to use instead.

Section A collects what is provable across nine areas, with the per-client federation gain
identified as the strongest result in the project (positive for every client, every seed, both
corpora; and a client missing classes still goes 0.6145 -> 0.8145). Section A9 lists the
**negative** results as contributions rather than apologies: leave-one-dataset-out at
0.0853/0.0763, the G1 run that passed vacuously on one malicious reading, and the analytical K
model mispredicting its own mechanism.

Section B is the guard rail -- eight claims that would be wrong, each with why and with the
honest replacement. The ones most easily slipped into: "detects attacks on a live agricultural
deployment" (the runtime classifies held-out records; no live extractor exists), "Ascon protects
the telemetry" (it protects the *weight channel*; the cloud hop is TLS by design), "validated on
Raspberry Pi hardware" (no hardware has run; container timings are not Pi timings), and "three
datasets were used" (three were evaluated, two were rejected, and the measurements are the
point).

Sections C and D list the ten figures with their data readiness and the two-column section
structure. Two things are outstanding: the sweep's final K = 50 points before VI-C can be
drafted, and the target venue and page limit, which decide how much of VI survives.

### G1 written up: the threat, the structural enforcement, and how strongly it was tested

`docs/g1-path-disjointness.md` (new) collects the project's central safety guarantee in one
place -- why a malicious reading must never reach the cloud, why the guarantee is enforced in the
object graph rather than by a coding rule, the four layers it is tested at, and the measured
results.

The enforcement argument: only `VerdictRouter` ever holds the cloud transport, and `AlertSink`
takes **no constructor parameters at all**, so there is no code path from a malicious verdict to
the cloud because the object handling malicious verdicts has nothing to reach it with. Breaking
G1 requires changing a constructor signature, which three tests fail on immediately.

The subtlety in the behavioural test is recorded explicitly: asserting `received_count == 0` is
not enough, because a system that transmits malicious payloads and has the receiver reject them
would also satisfy it. `rejected_count == 0` is what distinguishes "structurally unreachable"
from "sent and refused".

**The most useful finding is methodological, and it is a negative one.** The first containerised
run reported `G1: ... True` on **one** malicious reading, because the sensors publish before the
Pis finish federating and MQTT QoS 0 discards anything published before a subscription exists.
The assertion passed and nothing was learned: a guarantee about a class of events is only tested
as strongly as the number of events that occurred. Moving the attack window inside the Pis'
post-federation listening period took the count 1 -> 491 -> 4,239.

The document also separates two properties that fail for different reasons and are easy to
confuse -- the security property (no malicious reading reaches the cloud) and the delivery
property (every benign reading arrives) -- and states plainly what G1 does **not** cover: verdict
correctness. A misclassified attack goes to the cloud and G1 is still satisfied, which is why FPR
is reported first in every evaluation.

Results table, oldest to newest: Phase 7 end-to-end 55 malicious, 2026-09-19 loopback 119, the
vacuous twin attempt 1, then 491, then **4,239 malicious with 0 reaching the cloud and all 8,924
benign delivered**, cross-checked against the receiver's own independent count.

### A start-to-finish walkthrough of the software twin

`docs/software-twin-walkthrough.md` (new), written to be read by someone who has not seen the
topology. The user asked what the twin actually is and what was done at every step, so this
explains the reasoning and not only the result:

- **Why the twin exists at all** -- if you wire up two Pis and it fails, you cannot tell whether
  the fault is the code, the wiring, the Wi-Fi, the keys, the topics or the certificate, and
  every test cycle then costs a reflash on a board 10-20x slower than the laptop. Fixing
  everything that can be wrong in software first means a hardware failure is unambiguously a
  hardware failure.
- **What "twin" means**, and what it is not: same code (`asa aggregator`, `asa client-node`,
  `asa pi-runtime`, no test doubles), separate OS processes, a real network, and `linux/arm64`
  images so the aarch64 PyTorch wheels a Pi would install are the ones exercised. What changes on
  hardware is only *where the processes run*.
- **Why no single tool does this**: a Pi is a computer running Linux, so simulating one means
  emulating ARM Linux (QEMU, or Docker at process level), while an ESP32 is a microcontroller and
  simulates in a browser. Wokwi's Raspberry Pi Pico is a microcontroller and is not a substitute.
- **All ten containers individually** -- one image, nine roles selected by the compose command --
  including why there are two brokers rather than one (the farm boundary), why `sim-3` exists
  (K = 3 without a third Pi, because K = 2 federation is degenerate), and why `pi-1`/`pi-2` chain
  two commands (train periodically, classify continuously, which is the real deployment shape).
- **The twelve-step execution sequence**, including round 0 being the scaler round where only
  count/mean/M2 sufficient statistics leave a client, so the federated scaler equals a pooled one
  without pooling raw data.
- **The G6 boundary stated plainly**: the GRU never sees the sensor JSON; it classifies a
  held-out CICIoT2023 record paired to each message, and the scenario switch selects which
  held-out pool rather than synthesising anything. What that demonstrates is architectural
  correctness, not detection on a live agricultural deployment.
- **The verified R = 20 run**: 120 sealed frames, 0 rejected; 13,253 readings; 8,924 benign
  delivered with 0 failures; **4,239 malicious alerted locally with 0 reaching the cloud**, cross-
  checked against the receiver's own independent count of 8,924. Plus how strongly G1 has been
  tested over time (119, then an invalid 1, then 491, now 4,239).
- **The two invalid runs**, the rule behind the passing configuration, and the `init-secrets`
  trap that makes `--abort-on-container-exit` destroy the run.

### Breadboard wiring, wire by wire

`docs/breadboard-wiring.md` (new). The user asked for every individual wire and where it goes,
and for the platform question: the ESP32 simulator they were thinking of is **Wokwi**, which
this project already ships a diagram for. There is deliberately no equivalent for the Pi, and
the document says why -- an ESP32 is a microcontroller and can be simulated in a browser, while
a Raspberry Pi is a computer running Linux, so the only options are Docker (what the twin
already does, with the real codebase on a real network) or QEMU for a full OS image. Wokwi's
"Raspberry Pi Pico" is a microcontroller and does not run Linux, so it is not a substitute.

Contents: the eight wires as a table and as breadboard art, with only two of them signal wires
(GPIO 15 for DHT22 data, GPIO 34 for the moisture probe); the alternative layout with a 10 kOhm
pull-up for anyone who buys a bare AM2302 instead of the 3-pin module; the per-board firmware
table for all six nodes; and two hardware facts that are easy to get wrong -- **GPIO 34 must
stay on ADC1 because ADC2 stops working entirely while Wi-Fi is active**, and the DevKitC V4 is
wide enough to leave only one usable hole per pin row on a single breadboard.

It also records the known gap in simulating the sensor hop: Wokwi's free gateway reaches the
public internet but not a LAN, so a Wokwi board cannot reach a broker on the laptop or Pi. The
documented workaround is to meet on a public broker, which is a simulation convenience only --
on the real farm the broker runs on the Pi and readings never leave the farm on that hop.

### The registry now matches the documented decision, and the architecture has a diagram

**CICIoMT2024 is registered `not_for_training`.** The decision had been documented in
`docs/dataset-selection.md` while `TRAINING_CORPORA` still evaluated to
`("ciciot2023", "ciciomt2024")` -- and that tuple is the **default value** of
`run_phase9.py --corpora`, so any run started without an explicit flag would have trained on the
rejected corpus silently. The registry and the document now agree, and `tests/test_datasets.py`
asserts `TRAINING_CORPORA == ("ciciot2023",)` with every exclusion required to carry a reason
string, so they cannot drift apart again without a test failing.

Nothing was deleted. Both excluded corpora stay registered, characterisable
(`asa characterize --dataset ...`), and usable as ablations by naming them explicitly
(`--corpora ciciot2023,ciciomt2024`); `run_phase9.py` skips the leave-one-dataset-out folds
automatically when only one corpus is in play. `configs/generalised.yaml` records that its
candidate set is now the same as `default.yaml`'s, and why it is kept anyway.

**`docs/architecture-diagram.md`** (new) puts in one place what was previously spread across
plan prose, a Wokwi JSON file and the twin's compose file:

- the logical topology, with the cryptography named on each hop -- Ascon-AEAD128 both ways on
  the weight channel, plain TLS on the Pi -> cloud hop, plain MQTT farm-locally, and nothing at
  all on the malicious path because it never leaves the Pi;
- the **two planes** and the G6 boundary drawn explicitly, including which cache split supplies
  training rows (1,237,911) versus what the model classifies at runtime (311,565 held-out), and
  the statement that the ESP32 payload is never parsed into features;
- the **physical wiring**, read from `firmware/esp32_sensor/diagram.json` rather than
  transcribed: DHT22 data on GPIO 15, soil moisture on GPIO 34 (ADC1, because ADC2 is unusable
  while Wi-Fi is active), with wire colours and the capacitive-sensor requirement;
- the **twin mapping**, so each container is traceable to the hardware it stands in for, with
  the reminder that container timings are never Pi timings;
- the free tool list, which is Docker for the twin and Arduino IDE plus three libraries for the
  boards.

### Results write-ups for the twin and for Edge-IIoTset, and the R = 20 twin run

Two `results/` documents, in the folder's existing shape (what the architecture requires /
what we achieved / decisions flagged / honest gaps), so the paper is written from traceable
figures rather than from memory:

- **`results/phase8_software_twin.md`** — the containerised verification: 0 sealed frames
  rejected in either direction at 407,752 B up / 407,758 B down per round, and G1 holding
  across 491 malicious readings with 793 benign delivered and 0 malicious reaching the cloud,
  cross-checked against the receiver's own count. It also records the two *invalid* attempts
  and the rule behind the passing configuration, since a security property (no malicious
  reading reaches the cloud) and a delivery property (every benign reading arrives) fail for
  different reasons and only the first is a G1 claim.
- **`results/agri_native_edge_iiotset.md`** — the 0.9822 result, why the 8-class 0.7453 must
  not be quoted (DoS and Mirai are absent, so an 8-class macro average is capped near 6/8),
  the 26-column feature rule, the three classes excluded by measured degeneracy, and the two
  reasons the corpus is still not in the product model. The decisive one is that its unique
  contribution is **a single attack type**: against CICIoT2023's 34 leaf classes, only
  Ransomware is genuinely new — `Backdoor_Malware`, `MITM-ArpSpoofing`, `SqlInjection`, `XSS`,
  `Uploading_Attack`, `DictionaryBruteForce` and `Recon-*` are all already present. An earlier
  claim in this changelog that Backdoor and MITM were unique to Edge-IIoTset was wrong.

**A twin run at the demo's R = 20 is in progress** (120 sealed frames rather than 12), launched
alongside the K sweep at `SEQ_CAP=10000` — the crypto claim scales with rounds, not with
sequences, so the smaller cap costs nothing and leaves CPU for the sweep. `results/README.md`
gains rows for both documents and its test count is corrected to 438.

## 2026-09-30

### K sweep re-run for the paper, and K = 3 restated as a constraint rather than a result

The user asked why K = 3 was being treated as fixed. It is an inherited assumption, not a
measurement: `artifacts/k_threshold_analysis.json` carried the note "K is fixed at 3 by
assumption A1 / objective O3", Phase 4 used it because the paper did, and the hardware spec
gives it (two Pis plus one laptop-simulated client). Cancelling the empirical sweep earlier on
the grounds that "K = 3 is fixed" was circular — that sweep is precisely what tests the
assumption. The user's call: **hardware stays K = 3, but the results are reported over a K
sweep**, which is the stronger paper claim.

- **`scripts/run_k_sweep.py` re-launched over K in {3, 5, 10, 20, 30, 40, 50} x 3 seeds** at
  Phase 4's budget on `artifacts/phase4_cache.npz`. That cache is CICIoT2023 with the 16
  canonical features, which after today's corpus decision is exactly the headline design, so
  the sweep now measures the shipped model rather than a superseded configuration.
- **K = 40 added to the documented set.** The old set jumped 30 -> 50, straddling both the
  pigeonhole ceiling (42, set by BruteForce having only 42 blocks) and the predicted knee
  (K* = 48) without sampling either side. A point just below the ceiling is where the rare-class
  starvation should first appear.
- **Partial results are now written after every (K, seed) point** to
  `artifacts/k_sweep_results_progress.json`. The script previously wrote only on completion, so
  an interruption in a ~20 h run discarded everything; the first launch was in fact killed
  mid-run and its single completed point (K=3, seed=0, macro-F1 0.8507, 0 empty clients,
  3,560 s) survived only in a copied log.
- **The manifest note was corrected** to state that K = 3 is the hardware configuration chosen
  by budget, and because K = 2 federation is degenerate, rather than an optimum.

### Smart agriculture fixed as the domain, and Edge-IIoTset re-measured on its own schema

The user fixed **smart agriculture as the deployment domain**. Under the standing
domain-coherence principle that decides which corpora are admissible, not just how the project
is described: CICIoMT2024 is a *medical* testbed, and Edge-IIoTset is the only registered corpus
captured on agricultural equipment (soil-moisture, temperature, pH and water-level sensors over
MQTT, plus Modbus TCP).

`scripts/run_agri_native.py` (new) is the half-day centralised check the 2026-09-23 entry
scoped. It trains the GRU on Edge-IIoTset's **native** columns rather than on reconstructed
CICIoT2023 window features. Manifest `artifacts/manifest_agri_native.json`, 3 seeds, 20 epochs,
38.7 min.

**Result: macro-F1 0.9822 ± 0.0043 over the 12 viable native classes** (balanced accuracy
0.9882, MCC 0.9853, FPR 0.0017). Per class, everything ≥ 0.944, including the three families no
other registered corpus contains: **Ransomware 0.9440, Backdoor 0.9877, MITM 1.0000**. The
2026-09-21 figure of 0.12 measured our translation into the canonical 16-feature language, not
this corpus — that is now settled by measurement rather than by re-reading a report.

The 8-class `family` taxonomy scores 0.7453 and **must not be quoted**: Edge-IIoTset contains no
`DoS` and no `Mirai`, so both score 0.0000 in an 8-class macro average while the six families
present score 0.9882–0.9997. The mapping also collapses Ransomware, Backdoor, SQL_injection,
Uploading and XSS into one `WebBased` class, discarding exactly the coverage the corpus is
wanted for, which is why the native taxonomy is the reported one.

What the check had to establish first, and what it found:

- **The feature set is 26 behavioural columns**, not 63: the spec's `drop_columns`, minus 4
  zero-variance columns, minus 5 text columns, minus 9 identifiers (`tcp.seq`, `tcp.ack`,
  `tcp.ack_raw`, `tcp.checksum`, `icmp.checksum`, `icmp.seq_le`, `icmp.transmit_timestamp`,
  `udp.stream`, `mbtcp.trans_id` — transport nonces and capture-local counters). What remains is
  8 MQTT fields, 2 Modbus TCP fields, and TCP/DNS/ARP/ICMP/HTTP flags, lengths and timings.
- **Two text columns are worse than the known placeholder artefact.** All five carry the
  documented `"0"`/`"0.0"` provenance collision, but a whole-file census also showed
  `http.request.version` holding injected attack payloads
  (`-al&_PHPLIB[libdir]=http://cirt.net/rfiinc.txt?? HTTP/1.1`) and `dns.qry.name.len` holding
  DNS names rather than lengths — a column shift in the corpus build. Encoding either lets the
  model read the class off the payload. A numeric-only set excludes the artefact structurally.
- **Three classes cannot be expressed at all and are excluded by measurement, not by hand.**
  `--min-distinct-windows` (default 500) drops any class contributing fewer distinct windows:
  **DDoS_UDP contributes exactly 1, DDoS_ICMP 38, Fingerprinting 347.** Verified on raw strings:
  every one of the 26 features is literally zero for every UDP and ICMP flood packet, so those
  two classes are not even separable from each other. Their only distinguishing content lived in
  the dropped nonces. `frame.time` cannot rescue them either — the corpus build stripped month
  and day (`' 2021 11:44:10.081753000 '`), and DDoS_UDP timestamps do not parse at all.
- **R3 moved to window granularity, because the corpus is packet-granularity.** Row-level dedup
  deletes the repetition that *is* a flood; the record the model consumes is a 16-packet window.
  The gate now dedups windows and drops any test window whose content appears in train —
  ~10,400 per seed, real leakage that would otherwise have inflated the score. `--dedup-level`
  keeps the row-wise rule as an ablation.

**Stated limit:** 434,553 rows hold only **1,791 distinct packets** (99.6% duplicate). No
identical window spans the split, but train and test draw on the same packet vocabulary and
differ only in ordering. 0.9822 is within-corpus class separation, not cross-testbed
generalisation, and no leave-one-dataset-out check is possible because the feature language
differs from the other two corpora. Edge-IIoTset's `not_for_training` registration still stands
and is unchanged by this entry; re-registering it is a separate decision.

**Architecture consequence.** The three corpora cannot form one model. Edge-IIoTset's value is
in `mqtt.*`/`mbtcp.*` fields absent from CICIoT2023's extractor output, and FedAvg requires an
identical input width across clients, so a 26-input GRU cannot be averaged with a 16-input one.
Forcing the translation is what produced 0.12. The two models are therefore separate.

### Phase 8 software twin: sealed federation and G1 verified under load, in containers

`docker compose up` on the full topology (aggregator, 2 Mosquitto brokers, TLS cloud receiver,
pi-1, pi-2, sim-3, 2 sensor farms). All services exited 0.

- **Sealed federation:** scaler round + 2 weight rounds across 3 clients, **0 frames rejected**
  in either direction. Wire cost **407,752 B up / 407,758 B down per round**; scaler round
  1,480 B / 1,462 B. These are the per-round figures the hardware phase needs.
- **G1 under load:** pi-1 received 688 readings (470 benign → cloud, 0 failed; **173 malicious →
  173 local alerts**); pi-2 received 686 (323 benign → cloud, 0 failed; **318 malicious → 318
  alerts**). Cloud receiver **accepted 793 = 470 + 323, rejected 0**, and **0 malicious readings
  reached the cloud**.

**Bill of materials added** as `docs/plans/phase8-hardware-federation.md` §7a, so procurement
is version-controlled rather than reconstructed from the plan prose: 2x Pi 5 8GB, 6x ESP32
DevKitC V4, 6x DHT22 (GPIO 15) and 6x capacitive soil sensors (GPIO 34) with the pins taken from
`firmware/esp32_sensor/diagram.json`, plus power, storage and the LAN. Three purchasing-time
constraints are called out: Pi OS Bookworm is unusable (Python 3.11 against a >= 3.12 floor),
soil sensors must be capacitive rather than resistive, and bandwidth is a non-issue at the
measured 407 KB per client per round.

**The first attempt of this run was not a valid test and is recorded so it is not repeated.**
With the default `MESSAGES=240`, each Pi saw only **1** malicious reading, because the sensors
begin publishing while the Pis are still federating (~160 s) and MQTT QoS 0 drops anything
published before subscription — G1 "passed" on a single event. Raising `MESSAGES` to 800 without
raising `CLOUD_SECONDS` then produced a second false alarm: 190 benign sends "failed" only
because the receiver's 600 s budget expired mid-run while the Pis kept sending. Both the attack
window and the receiver window must cover the Pi's *post-federation* listening period; the
passing configuration is `MESSAGES=800 ATTACK_AFTER=400 CLOUD_SECONDS=2400`.

### Every farm is better off federating: the mixed-layout federation-gain baseline

The `--local-only` flag added on 2026-09-23 had no numbers attached to it. It does now. The run
finished 2026-09-23 22:00 after **12.3 hours** (44,172 s) and sat uncommitted for a week;
manifest `artifacts/manifest_phase9_federation_gain_mixed.json`, log
`artifacts/phase9_federation_gain_mixed.log`. This is Section III-I1's **baseline 4** (gap G4)
for the mixed-farm layout — Phase 4 measured it on one corpus, and the mixed layout had no such
baseline until now.

Each farm was trained on its own data alone, compute-matched to the federated run at R × E = 60
local passes, and scored on both corpora's test splits. 3 seeds, present-only macro-F1:

| farm | local-only CICIoT2023 | federated | gain | local-only CICIoMT2024 | federated | gain |
| --- | --- | --- | --- | --- | --- | --- |
| farm-0 | 0.6873 | 0.8211 | **+0.1338** | 0.7146 | 0.8714 | **+0.1568** |
| farm-1 | 0.7422 | 0.8211 | **+0.0789** | 0.7167 | 0.8714 | **+0.1547** |
| farm-2 | 0.7695 | 0.8211 | **+0.0516** | 0.6744 | 0.8714 | **+0.1970** |

Federated global: **0.8211 ± 0.0153** on CICIoT2023 (FPR 0.3269 ± 0.0873) and
**0.8714 ± 0.0118** present-only on CICIoMT2024 (FPR 0.0143 ± 0.0005).

The gain is positive for all three farms on both corpora on every seed, so the mixed layout now
has the per-client justification the corpus-per-client layout never earned. It is a stronger
result than Phase 4's single-corpus version (+0.1023 / +0.1565 / +0.0721) in one specific way:
the gain holds across two testbeds at once, which is what a farm operator actually buys by
joining. The spread across farms tracks the Dirichlet draw as expected — farm-0 holds the most
lopsided CICIoT2023 share and gains most there, farm-2 the weakest CICIoMT2024 share and gains
most there.

Read the CICIoT2023 FPR of 0.33 first, as in Phase 4: macro-F1 is carried by the attack families
and the Benign class stays the weak one (per-class F1 0.6843).

## 2026-09-23

### Phase 9 third design: a farm models a deployment site, not a corpus

The user objected to the second run's client layout: allocating K = 3 as two CICIoT2023 farms
plus one CICIoMT2024 farm does not model a real deployment, because detection for a medical
setting federates over medical sites and one for agriculture over farms. A client split that
encodes which corpus a client came from is a cross-domain consortium, which this project is not
building. The run's own numbers agreed — on the CICIoT2023 test set, one corpus federated scored
0.8407 and both corpora pooled scored 0.8390, but both corpora split corpus-per-client scored
only 0.7742, so the second corpus as *data* cost nothing while the second corpus as its own
*client* cost 0.066.

- **`run_phase9.py --client-allocation mixed`** (now the default): every farm draws a Dirichlet
  share of every training corpus, each corpus partitioned under its own derived seed
  (`seed * 1000 + corpus_index`) so the draws are independent. All three farms hold 8/8 families
  where the CICIoMT2024-only farm held 6/8. `per-corpus` is kept as the labelled cross-domain
  ablation. `build_farms` returns one representation for both layouts (`{corpus: row_mask}` per
  farm) so the rest of the pipeline is layout-agnostic; windows are built inside each corpus's
  own rows and only then concatenated, and per-corpus scaling is unchanged because a mixed farm
  contributes one set of sufficient statistics per corpus it holds.
- **`--run-name`, and the layout and scaling flags folded into the derived manifest name.** A
  smoke run overwrote the 8.82 h result's manifest, which was recovered from git; each
  configuration now writes to its own file.
- **`--local-only`**: trains each farm on its own data alone, compute-matched to the federated
  run at R x E epochs and scored on the same test sets, so the per-client gain answers whether a
  farm is better off federating (III-I1 baseline 4, gap G4). Phase 4 measured this on one corpus
  (+0.1023 / +0.1565 / +0.0721 per client, 82.4 % of the local-to-centralised gap recovered);
  the mixed layout had no such baseline until now.

Results, 3 seeds at Phase 4's budget, manifest
`artifacts/manifest_phase9_generalised_percorpus_mixed.json`:

| experiment | test corpus | per-corpus farms | mixed farms |
| --- | --- | --- | --- |
| in_distribution | CICIoT2023 | 0.7742 ± 0.0119 | **0.8211 ± 0.0153** |
| in_distribution | CICIoMT2024 (present) | 0.8034 ± 0.0207 | **0.8714 ± 0.0118** |
| pooled_centralised | CICIoT2023 | 0.8390 ± 0.0035 | 0.8476 ± 0.0048 |
| lodo/ciciot2023 | CICIoT2023 **held out** | 0.0858 ± 0.0347 | 0.0853 ± 0.0288 |
| lodo/ciciomt2024 | CICIoMT2024 **held out** | 0.1380 ± 0.0485 | 0.0763 ± 0.0251 |
| lodo/ciciomt2024 | CICIoT2023 (ceiling) | 0.8407 ± 0.0150 | 0.8443 ± 0.0003 |

Adding the second corpus cost 0.066 against the single-corpus ceiling and now costs 0.020, so
about 70 % of the penalty was the layout. The price of federating rather than pooling fell from
0.065 to 0.027. Leave-one-dataset-out did not move and was not expected to: the layout governs
how clients are built, not whether window statistics transfer across testbeds. At seed 0 the
single-corpus LODO folds reproduce the second run to four decimals, because with one training
corpus both layouts construct the same three Dirichlet farms.

### Edge-IIoTset: the corpus is clean, our reconstruction was not

Re-reading `artifacts/phase1_characterization_report_edge_iiotset.json` while scoping a possible
domain-specific rebuild contradicted the figure this project dropped it on. The 81 % duplication
was a property of **our packet-field reconstruction** of CICIoT2023's 16 window features, not of
the corpus: the raw file has **815 exact duplicates in 2,219,201 records (0.04 %)**, an imbalance
ratio of 1614 against CICIoT2023's 5751, and 63 native columns including ten MQTT fields and
three Modbus TCP fields — the protocols farm automation runs on. Its 14 attack types cover what
CICIoT2023's 8 families cover and add ransomware, which neither other corpus has. CICIoMT2024 by
contrast is network-layer only: floods, scans and ARP spoofing, with no malware, no web attacks
and no brute force, so it cannot stand alone as a domain corpus.

No decision taken: the `not_for_training` registration stands, and the reason string remains
accurate about the reconstruction. Recorded because a domain-specific rebuild on Edge-IIoTset's
*native* schema is now a live option, gated on a half-day centralised check.

## 2026-09-22

### Phase 9 second design: Edge-IIoTset out of training, 16 features back, per-corpus scaling, pooled twin

The user read the 2026-09-21 run (macro-F1 0.83 → 0.60 on CICIoT2023, leave-one-dataset-out
≈ 0) and asked whether the two added corpora were a good fit. The answer, from the manifests
and `artifacts/phase9_cache.npz` rather than from memory: the 0.83 → 0.60 drop stacks four
changes (Phase 3 complete 0.830 → Phase 3 default driver 0.787 → 13-feature intersection
0.763 → three-corpus federation at R = 10, E = 2, 150k/client 0.603), so the feature cut
cost 0.024 and the rest was corpus mixing plus an unmatched budget; LODO ≈ 0 with FPR 0.6–0.99
is a scale mismatch, not a generalisation number. CICIoMT2024 is the right second corpus
(same extractor family, all 16 features, MQTT floods) once its per-corpus scale is handled;
Edge-IIoTset is not usable as a network-feature corpus in this form. Decided, implemented:

- **`data/datasets.py`:** `DatasetSpec.not_for_training` (a reason string) and
  `TRAINING_CORPORA` (every registered corpus without one). Edge-IIoTset carries the measured
  reason: near-constant reconstructed windows (99 % TCP, 93 % ACK-flagged), 81 % exact
  duplicates, in-distribution 0.12 macro-F1 / 0.93 FPR on its own split, and the only corpus
  that cannot supply `Header_Length`, `IAT`, `Time_To_Live`. It stays registered and
  characterisable; `tests/test_datasets.py` pins the training set to CICIoT2023 + CICIoMT2024.
- **`configs/generalised.yaml`:** the candidate set is now the full 39-column canonical
  vocabulary (the CICIoT2023 ∩ CICIoMT2024 intersection), so all 16 Phase 2 features are
  available again. The 17-column record of the three-corpus run stays in
  `artifacts/manifest_phase9_generalised.json`.
- **`scripts/run_phase9.py`** (rewritten): K = `federated.n_clients` allocated across the
  training corpora (larger corpus gets the extra farm; Phase 4's block Dirichlet inside a
  corpus; three Dirichlet farms for a single-corpus fold) with client ids `corpus:index`;
  `--scaling per-corpus` (default; each corpus's farms combine their sufficient statistics
  into that corpus's scaler, a held-out corpus uses its own training split's label-free
  statistics) vs `global` (III-F4, the ablation); feature selection on the scaled rows; a
  `pooled_centralised` twin trained on the concatenation of the *same* capped farm windows
  for R × E epochs (same sequences, no federation); LODO folds scored on the training
  corpus's own split too (the single-corpus ceiling at the same budget); seed-major order
  with `artifacts/<run>_progress.json` rewritten after every run; the cache now carries
  block ids and refuses a cache built for a narrower candidate list.
- **`docs/plans/phase9-multi-dataset.md` §5** records the design and the stated expectation
  before the run. The one third-corpus candidate, CIC IoT-DIAD 2024, was checked the same
  day: its packet-based set is the IoTDevID per-packet schema and its flow set is
  CICFlowMeter, so none of the 39 canonical columns appears -- no drop-in third corpus exists
  in CICIoT2023's feature family, and the training set is the two.

Run completed in 8.82 h; manifest `artifacts/manifest_phase9_generalised_percorpus.json`,
writeup [`results/phase9_multi_dataset.md`](results/phase9_multi_dataset.md). Dedup-before-split
held independently on both corpora (`r3_overlap = 0`). 3 seeds, mean ± std:

| experiment | test corpus | macro-F1 | present-only | FPR |
| --- | --- | --- | --- | --- |
| in_distribution | CICIoT2023 | 0.7742 ± 0.0119 | 0.7742 ± 0.0119 | 0.3183 ± 0.0182 |
| in_distribution | CICIoMT2024 | 0.6025 ± 0.0155 | 0.8034 ± 0.0207 | 0.1188 ± 0.1018 |
| pooled_centralised | CICIoT2023 | 0.8390 ± 0.0035 | 0.8390 ± 0.0035 | 0.2929 ± 0.0330 |
| pooled_centralised | CICIoMT2024 | 0.6757 ± 0.0039 | 0.9009 ± 0.0052 | 0.0242 ± 0.0011 |
| lodo/ciciomt2024 | CICIoT2023 (ceiling) | 0.8407 ± 0.0150 | 0.8407 ± 0.0150 | 0.2320 ± 0.0458 |
| lodo/ciciomt2024 | **CICIoMT2024 held out** | 0.1035 ± 0.0364 | 0.1380 ± 0.0485 | 0.8174 ± 0.0964 |
| lodo/ciciot2023 | CICIoMT2024 (ceiling) | 0.6778 ± 0.0058 | 0.9038 ± 0.0077 | 0.0095 ± 0.0060 |
| lodo/ciciot2023 | **CICIoT2023 held out** | 0.0858 ± 0.0347 | 0.0858 ± 0.0347 | 0.9520 ± 0.0433 |

Three of the four comparisons came out as designed and the headline one did not:

- **The rewritten driver reproduces Phase 4.** Trained on CICIoT2023 alone at the same budget
  it scores 0.8407 ± 0.0150 against Phase 4's 0.8308 ± 0.0150 — within a seed's spread, on a
  wider candidate vocabulary and a per-corpus scaler. The Phase 9 rework broke nothing.
- **Per-corpus scaling fixed the in-distribution collapse**, which is what the second design
  was built to test: CICIoT2023 0.603 → 0.774, CICIoMT2024 present-only 0.562 → 0.803. The
  **pooled two-corpus twin reaches 0.8390 ± 0.0035, matching Phase 3's single-corpus 0.830** —
  a model that sees both corpora at once pays nothing for the second.
- **Mixing costs ≈ 0.066 and federation ≈ 0.065** on CICIoT2023 (ceiling 0.841 → federated
  two-corpus 0.774 → pooled 0.839).
- **Leave-one-dataset-out failed.** 0.086 on held-out CICIoT2023 (FPR 0.95) and 0.138
  present-only on held-out CICIoMT2024 (FPR 0.82), against 0.84 / 0.90 on the corpus each model
  trained on: the detector flags nearly every benign window of a testbed it has not seen. The
  plan's pre-registered expectation ("well below, but no longer near zero") was **not met**.
  Per-corpus standardisation removed the scale offset blamed for the first run's LODO ≈ 0 and
  the number did not move, so that explanation was incomplete; the residual gap is shift in the
  features themselves. Reported, not tuned away (R4, III-J). What Phase 9 can claim is
  generalisation across farms and across two jointly-seen testbeds, **not** transfer to an
  unseen one; cross-testbed transfer needs domain adaptation or corpus-invariant features,
  neither in scope.

## 2026-09-21

### Phase 9: CICIoMT2024 and Edge-IIoTset are on disk; two characterisation bugs fixed on first contact

The two new corpora landed (both gitignored under `data/`): CICIoMT2024 `WiFI_and_MQTT/attacks/
CSV/{train,test}` complete (51 + 21 files, every file checked to end with a newline; not
`profiling/`, `PCAP/` or `Bluetooth/`, which the plan and the registry exclude), and
Edge-IIoTset `DNN-EdgeIIoT-dataset.csv` (63 columns, 1.2 GB). Running `asa characterize
--dataset edge_iiotset` on real data, the first time the Phase 9 scan met a corpus that was not
synthetic, exposed two defects in the scan itself:

- **Per-chunk dtype inference broke cross-chunk reductions.** `pd.read_csv(chunksize=...)`
  infers each column's dtype per chunk, so Edge-IIoTset's `arp.dst.proto_ipv4` (the placeholder
  `0` for most rows, an IP address for the rest) came back `int64` from one chunk and `object`
  from the next, and the cross-chunk `min()` died with `TypeError: '<=' not supported between
  instances of 'str' and 'int'`. Had the string rows come first the float moment block would
  have died instead. `characterize_dataset` now runs a probe pass (`_text_columns`) that
  settles every column's type corpus-wide -- text iff any non-null value fails to parse as a
  number -- and reads those columns as `str` in the real scan, so the report no longer depends
  on where a chunk boundary falls. The committed CICIoT2023 reports are unaffected (no column
  there is mixed).
- **A placeholder collision split across chunks was invisible.** `placeholder_collisions` only
  flagged a column when both spellings occurred *within one chunk*, and the scan unioned those
  per-chunk verdicts -- so `"0"` in one source file and `"0.0"` in another, which is exactly
  the file-provenance pattern the check exists to catch, went unreported. New
  `datasets.placeholder_spellings` returns what each chunk contains; the scan unions it over
  the corpus and judges the collision at the end. `placeholder_collisions` is now a thin
  wrapper over it and keeps its contract for single frames.
- `tests/test_characterize.py`: a regression test with a mixed `0`/IP/`0.0` column and a
  two-row chunk size covering both defects.

Both corpora characterised on the fixed scan (reports at
`artifacts/phase1_characterization_report_{ciciomt2024,edge_iiotset}.json`) and reconciled with
the registry — plan §3 step 1 is done; the findings are recorded in the plan. What the real
file changed in code:

- **CICIoMT2024 is the drop-in the plan predicted:** 19/19 labels in the taxonomy, 16/16
  selected features through `harmonise_columns`, no text columns, no collisions.
- **Edge-IIoTset's placeholder artefact is real and neutralised:** 17 columns collide on
  `"0"`/`"0.0"` over the raw file; 0 after `normalise_placeholders`, verified over all 2.2 M
  rows (the file is class-sorted, so nothing short of a full scan proves it).
- **`packet_windows.edge_iiotset_packets`, reconciled against the file:** (i) a missing
  `tcp.flags` (a normalised placeholder on every non-TCP packet) reached the hex parser as a
  float under pandas 3 and crashed; parsed with `na_action="ignore"` now. (ii) `udp.port` is
  the placeholder on every `DDoS_UDP` row; `udp.stream` is what marks a UDP packet, so the
  derivation now takes either. (iii) **`frame.time` is the literal `"6.0"` on every `DDoS_UDP`
  and `MITM` row** — two classes with no time base, so `IAT`/`Rate` would be `NaN`/`0` for
  exactly those windows, a class marker. User decision: drop the two *features* for this
  corpus, not the two classes (MITM exists in no other corpus). `PacketColumns.time_s` is now
  optional and `aggregate_packet_windows` emits no `IAT`/`Rate` without it; the harmoniser
  reports them missing. Edge-IIoTset supplies **12/16** selected features (`Header_Length`,
  `Time_To_Live`, `Protocol Type`, `IAT` missing; nothing imputed).
- Verified on a stratified 249 k-packet slice of the real file: normalise → packets → 24.9 k
  windows of 10 → harmonise, 0 NaN, 0 collisions, per-class TCP/UDP shares physically sensible.
- `tests/test_datasets.py`: two new tests (derivation after placeholder normalisation with the
  real `udp.port`/`udp.stream` pattern; aggregation without a time base reports `IAT` missing).

### Phase 9 step 4 first: the cost of the cross-corpus feature intersection, measured

The user's direction: the model must work *generally*, not just on CICIoT2023. The question
that decides the design is what restricting to the features every corpus supplies costs on
the corpus the project was built on. Answered before any federation code:

- `packet_windows.edge_iiotset_packets` now derives the **IP protocol number** (6/17/1 from
  the TCP, UDP and `icmp.checksum` markers, 0 for non-IP such as MITM's ARP), so `Protocol
  Type` (the window mode) is no longer missing for Edge-IIoTset: **13 of the original 16**
  selected features are cross-corpus.
- `features.candidate_columns` (config, `FeatureConfig`) + `selection.restrict_candidates`:
  a **Stage 0** that limits the universe the four stages select from. `None` = the
  single-corpus runs of Phases 2-7, unchanged. A listed column the corpus lacks raises, never
  imputed. Wired into `run_phase3`, `run_phase3_complete`, `run_phase4`, `run_phase7`.
- `configs/generalised.yaml`: `default.yaml` with the intersection as candidates. The
  intersection, computed from the real files: CICIoT2023 and CICIoMT2024 supply 39 canonical
  columns each, Edge-IIoTset's windows 18. `Number` is excluded (fixed at the window size for
  Edge-IIoTset by construction, so a testbed marker), leaving 17 candidates.
- **Ablation, `artifacts/manifest_phase3_generalised.json`** — Phase 3 centralised GRU on
  CICIoT2023, 3 epochs × 3 seeds, W=16, identical split (1,222,009 / 296,750 sequences),
  against `manifest_phase3_default.json`. Stage 2's Spearman prune collapses the 17 to
  **F = 13** (`AVG`≈`Tot size`, `Std`≈`Variance`, `*_count`≈`*_flag_number`):

  | metric | F=16 (default) | F=13 (intersection) | Δ |
  | --- | --- | --- | --- |
  | macro-F1 | 0.7866 ± 0.0199 | 0.7629 ± 0.0098 | −0.024 |
  | balanced accuracy | 0.8660 ± 0.0084 | 0.8380 ± 0.0045 | −0.028 |
  | MCC | 0.8532 ± 0.0097 | 0.8407 ± 0.0070 | −0.013 |
  | binary FPR | 0.2573 ± 0.0612 | 0.3059 ± 0.0757 | +0.049 |

  Per class, the loss is concentrated where the three dropped features carried the signal:
  BruteForce 0.51 → 0.40, Benign 0.70 → 0.67; the flood families are within a point. So the
  cost of generality on CICIoT2023 is **about 2.4 macro-F1 points, ~1.2 baseline σ** — real,
  but small next to what the intersection buys (MITM, injection, ransomware, MQTT floods,
  which CICIoT2023 cannot teach at all). Reported next to the generalisation results, never
  instead of them.

### Phase 9 steps 2-6: the generalised detector -- `asa generalise`

The user's direction, verbatim in spirit: the GRU must not be good on one dataset only, it
must be good *generally*. This entry is the build that measures that.

- `data/corpus.py` (new): `load_corpus(spec, ...)` -- one registered corpus in, one
  harmonised, leaf-labelled frame out (canonical columns the corpus actually supplies +
  `label` + `source_file`), ready for `dedup -> make_blocks -> stratified_block_split`
  unchanged. Window corpora go through the Phase 2 capped subsample (`subsample.py` gained a
  `label_from_path`/`parts` hook so CICIoMT2024's `_train`/`_test` filenames label correctly;
  the CICIoT2023 default is untouched); Edge-IIoTset goes text -> `normalise_placeholders`
  -> `edge_iiotset_packets` -> `aggregate_packet_windows` -> per-leaf cap on *window* rows,
  so the cap means "rows the model sees" for every corpus. `SequenceConfig.window_packets`
  (10) is the knob.
- `federated/server.py`: **`SealedFederatedServer`** -- the Phase 4 round with the Phase 8
  channel on every exchange: each client seals `(theta_k, n_k)` under its `k_up`, the
  aggregator opens it (a frame that opens to bottom is *dropped*: weight 0 that round), and
  seals the new global back per client under `k_down`; a client trains from what it last
  authenticated. `run_federation(..., sealed=(client_ids, keys))` selects it.
  `tests/test_sealed_server.py`: bit-for-bit equal to the plain server over 3 rounds; a
  tampered client's frames are dropped and the aggregate equals a federation without it.
- `scripts/run_phase9.py` / **`asa generalise`**: per corpus, load -> dedup -> block split
  -> **R3 gate** (raises on any shared hash); selection over `candidate_columns` fitted on
  the *training* corpora of each experiment only; global scaler from client sufficient
  statistics; **one corpus per client** (client id = corpus name); two experiments over
  >= 3 seeds -- `in_distribution` (K = 3, scored on every corpus's own test split) and
  `lodo/<held-out>` (K = 2, scored on the held-out corpus's test split, the same
  sequences). Reports `macro_f1` (C = 8, absent families score 0) **and**
  `macro_f1_present` (families present in that split -- CICIoMT2024 has no Mirai or
  BruteForce, Edge-IIoTset no DoS or Mirai). Budget flags `--rounds/--local-epochs/
  --sequence-cap`; `--cache` holds the prepared corpora so a re-run with another budget
  skips loading. `tests/test_corpus.py` covers the loader on synthetic files shaped like the
  real ones.
- Data facts from the real run, recorded in the manifest: CICIoT2023 1,772,371 -> 1,549,531
  after dedup; CICIoMT2024 924,291 -> 920,523; **Edge-IIoTset 130,351 windows -> 24,753
  (81 % exact duplicates)** -- with only `tcp.len` and flag bits per packet, flood windows
  are identical rows, so Edge-IIoTset is a small client by construction. R3 overlap 0 on all
  three.

**First full run, `artifacts/manifest_phase9_generalised.json`** (R = 10, E = 2, 150k
sequences/client, 3 seeds; Phase 4's budget was R = 20, E = 3, uncapped, so the
in-distribution numbers are a lower budget than Phase 4's 0.83 and the CICIoT2023 curve was
still rising at round 10). Macro-F1 over families present in the test split, mean ± std:

| experiment | test corpus | macro-F1 (present) | FPR |
| --- | --- | --- | --- |
| in_distribution (K = 3) | CICIoT2023 | 0.603 ± 0.001 | 0.19 |
| in_distribution (K = 3) | CICIoMT2024 | 0.562 ± 0.032 | 0.37 |
| in_distribution (K = 3) | Edge-IIoTset | 0.118 ± 0.022 | 0.93 |
| lodo (train MT+Edge) | CICIoT2023 | 0.064 ± 0.011 | 0.58 |
| lodo (train IoT+Edge) | CICIoMT2024 | 0.265 ± 0.056 | 0.92 |
| lodo (train IoT+MT) | Edge-IIoTset | 0.092 ± 0.022 | 0.99 |

**Reading, stated plainly.** (i) The federation trains: the two large clients reach usable
in-distribution scores under a reduced budget, and the sealed channel carried every frame
(0 dropped). (ii) The small client is swamped: Edge-IIoTset holds 19,372 of 319,373 training
sequences (6 % of the Eq. 21 weight) and the global model does not fit it -- 81 % of its
benign windows are called Spoofing. (iii) **Leave-one-dataset-out is near zero on every
fold, and the per-round curves are flat**, so this is not a budget problem: a model trained
on two testbeds does not recognise the third's traffic, benign included (FPR 0.58-0.99). The
"testbed shortcut" the plan warned about is not a risk, it is the dominant signal.
(iv) The cache shows why, feature by feature: the *same canonical column* lives on different
scales per corpus -- `ack_count` median 2 / 0 / 10 and std 27 / 0.3 / 1.4 across
CICIoT2023 / CICIoMT2024 / Edge-IIoTset; `Protocol Type` is integer protocol numbers in two
corpora and fractional values (0.1, 0.2, ...) in CICIoMT2024, i.e. a mean, not a mode;
benign `Tot sum` medians 1.8k / 1.1k / 6.5k. The "same extractor" claim for CICIoMT2024
holds for column *names* and not for column *values*. No harmonisation by renaming can fix
that, and the global (pooled-equivalent) scaler of III-F4 cannot either: it standardises the
union, so each corpus keeps its own offset and the model reads the offset as the label.
The negative result is the honest generalisation number and is recorded as such; the
follow-ups (per-client standardisation or rank features -- a design change to III-F4;
re-verifying CICIoMT2024's feature semantics against its README; the plan's stretch item,
one extractor over raw pcaps) are the user's call and are listed in the plan.

**Follow-up decided the same day** (user: "do what you think is best"): `run_phase9.py
--scaling per-client` -- each client standardises with its own training statistics, and a
test corpus with the statistics of its own training split (for a held-out LODO corpus, the
label-free feature statistics a new farm would compute over its own traffic before running
the model; no test row, no label). This departs from III-F4's single global scaler and the
paper absorbs it if the result is kept. Queued at the Phase 4 budget (R = 20, E = 3, 400k
sequences per client ≈ Phase 4's per-client load) behind the empirical K sweep; manifest
`manifest_phase9_generalised_perclient.json` when it lands.

### Phase 4 extension: the empirical client-count sweep, finally run

`scripts/run_k_sweep.py --k-values 3,5,10,20,30,50` at Phase 4's exact budget on the Phase 4
cache, so K = 3 reproduces `manifest_phase4_default.json` and every point is read against
the 0.8543 centralised ceiling. The analytical prediction it tests
(`artifacts/k_threshold_analysis.json`): K* ≈ 48 at α = 0.5, BruteForce pigeonholed past
K = 42, communication break-even with the paper's pooling figure at K = 51. Results below when
the run completes; an α ∈ {0.1, 100} pass around the knee follows.

## 2026-09-19

### Phase 8 Steps 1–2: the Ascon-sealed weight channel and networked federation nodes

The weight exchange between each local GRU and the master GRU is now Ascon-AEAD128 sealed in
both directions, and the federation runs as separate processes over TCP.

- `crypto/ascon_aead.py`: `WeightAssociatedData` `⟨client_id, round, direction,
  schema_version⟩` with its own `fmt_version` byte (0x02), so a telemetry AD and a weight AD are
  mutually unparseable by construction.
- `federated/transport.py` (new): `WeightSealer`/`WeightOpener` and the `ClientChannel` /
  `AggregatorChannel` pairs. Frame = `magic ‖ nonce ‖ L(ad) ‖ ad ‖ ciphertext‖tag`; plaintext is
  the existing safetensors blob, so `n_k` is authenticated too. One key per client per
  direction, so every key has exactly one encrypting process and the nonce registry's
  zero-reuse guarantee is structural. `open()` returns `⊥` for a bad tag, malformed frame,
  foreign client, wrong direction, wrong schema, or a round other than the one expected;
  a replayed round-r frame is dead once round r completes. Demo keys are written as a
  labelled JSON file that `load_keys` refuses without the label; `keys/` is gitignored.
- `federated/node.py` (new): `AggregatorNode` (threaded TCP server with a per-round barrier;
  a frame that fails to open is dropped with no reply and logged locally) and `ClientNode`
  (train → seal → send → open → repeat; retries a dropped connection with the same round's
  frame, which is safe because the frame is bound to that round). **Round 0 is the scaler
  exchange over the same channel**: each client seals `(count, mean, M2)`; the aggregator
  combines with Chan's formula and seals `(mean, std)` back. The initial global state is
  derived from the run seed on every node, never transmitted.
- `federated/cache_partition.py` (new): one client's unscaled partition from the Phase 4 cache
  with the shared Dirichlet draw, plus `sequence_cap` (capture order, no shuffle) for Pi-class
  compute — documented as a simulation convenience.
- `scripts/run_aggregator.py`, `scripts/run_client_node.py`; `asa aggregator`, `asa client-node`.
  Both refuse to run unless the Ascon KAT passes, and write manifests with per-round byte
  counts, timings and the rejection log.
- Tests (all gating): `test_weight_channel.py` (22: bit-exact roundtrip incl. `n_k`, FedAvg over
  the channel == in-process, single-bit tamper in every region → `⊥` with nothing applied,
  truncation/foreign protocol/wrong key, replay into same and next round, uplink reflected as
  downlink, foreign client, schema mismatch, 100 sealed frames with 100 distinct nonces, key
  file label enforcement) and `test_node_loopback.py` (2: three client threads + aggregator
  over loopback reproduce `FederatedServer` **bit-for-bit** including the sealed scaler round;
  a tampering proxy on the wire → one rejection, nothing applied, client recovers).
- Verified by hand, not committed as a result: four processes over loopback on the real
  `artifacts/phase4_cache.npz` (cap 5,000 sequences, R=2, E=1) — scaler round carried the true
  per-client row counts 297,067 / 473,352 / 467,492 (identical to Phase 4's manifest), each
  sealed weight frame was 135,917 B (≈ 0.05 % over the 135 KB plaintext, against 33 % for
  96-byte telemetry), zero rejections. The manifests were deleted so a smoke run does not sit
  in `artifacts/` looking like a finding.
- Measured while testing: the vendored pure-Python Ascon costs ≈ 0.5 s per 135 KB seal on
  this laptop. Fine per round; the Pi figure is a plan §7 deliverable.

### Phase 8 software twin: the whole hardware topology simulated before any hardware exists

The user was asked to simulate the entire setup in software first. No free online tool
simulates a Linux Pi and an ESP32 together, so it is split into three free layers (Wokwi for
the ESP32, Docker Compose for Pis/laptop/cloud, QEMU optional for the OS image) — see
`docs/plans/phase8-software-twin.md`. Docker is not installed on the development machine, so
the compose file is validated structurally and by a headless process-level run, not by
`docker compose up`; that run is the user's.

- **Sensor contract** (`telemetry/mqtt_source.py`, new): `farm/<farm>/sensor/<device>` JSON with
  a monotonic `seq`, plus `farm/<farm>/scenario`. Parses, orders, drops replays/malformed/
  foreign messages, tracks per-device scenario. Driven by an injected fake client in tests;
  `paho-mqtt` (new dependency) only for a real broker.
- **Scenario-restricted provenance** (`telemetry/provenance.py`): `network_features_for(...,
  scenario=)` draws from the benign-only or attack-only held-out pool with the same fixed
  permutation discipline; refuses (never synthesises) when no such records exist. G6 unchanged.
- **TLS benign path** (`routing/tls_path.py`, new): `ReadingEnvelope`, `TlsVerdictRouter`
  (only holder of the transport; alert sink never sees one), `HttpsCloudClient`,
  `CloudReceiverServer` (HTTPS or plain HTTP, per-device `ReplayGuard`, `/readings`, `/stats`).
  The Phase 6/7 Ascon-on-telemetry modules are untouched and their tests still run.
- **Scripts / CLI**: `asa pi-runtime` (MQTT or `--source sim` → provenance → windows → the
  global GRU the node received → Eq. 5 → TLS cloud / local alert; per-stage latency; manifest
  tagged `platform`), `asa cloud-receiver` (`--generate-cert` for a self-signed demo cert),
  `asa virtual-sensor` (the ESP32 contract from Python, with a scripted attack switch);
  `asa client-node` now saves its final model + scaler for the runtime and takes `--platform`.
- **Firmware** `firmware/esp32_sensor/` (`.ino`, Wokwi `diagram.json`, libraries, README):
  DHT22 + moisture probe → MQTT; subscribes to the scenario topic; readings never change.
- **Docker** `docker/Dockerfile` (python:3.12-slim, arm64 by default, CPU torch),
  `docker/mosquitto.conf`, `docker/init-secrets.sh` (one-shot demo keys + cert into volumes),
  `docker-compose.yml` (init-secrets, aggregator, cloud-receiver, two farm brokers, pi-1,
  pi-2, sim-3, two virtual-sensor services), `.dockerignore`.
- **Tests**: `test_mqtt_source.py` (7), `test_tls_path.py` (6, gating: malicious-only run
  emits nothing; alert sink holds no transport; real HTTP round-trip with replay rejection),
  `test_software_twin.py` (compose references only registered subcommands; topology matches
  the plan; `docker compose config` when Docker exists; headless end-to-end run when the
  Phase 4 cache exists — asserts cloud accepted == benign verdicts, 0 rejected, alerts ==
  malicious verdicts, and the scenario switch reached the adapter).
- Verified by hand on loopback (`platform=laptop-loopback-smoke`, manifests deleted
  afterwards): 1 sealed round with 3 clients, 136,410 B/frame, 0 rejections; then 240
  readings with an attack switch at 120 → 76 benign → cloud accepted 76, 119 malicious → 119
  alerts, 0 at the cloud.

### `scripts/fetch_ciciomt2024.py`: resumable, selective download of the WiFi/MQTT CSVs

The user's connection is slow and the UNB listing has one link per file (the TCP/IP floods
alone are ~40 numbered parts). The script reads the listing page, keeps every un-numbered
file plus **part 1** of each numbered flood family (the subsampler's per-class cap means
parts 2+ would never be read), and downloads with ``curl -C -`` so an interrupted transfer
resumes. ``--all`` fetches everything; ``--dry-run`` lists. The listing's filenames
(``TCP_IP-DDoS-ICMP1_train.pcap.csv``) confirm the spelling the taxonomy and
``label_from_ciciomt2024_filename`` were written for. Test: selection rule + every kept name
resolves to a known leaf.

### Phase 9 scaffolding: multi-dataset generalisation (data not yet downloaded)

The user lifted the "CICIoT2023 only" rule: the detector must generalise across two to three
IoT intrusion corpora. Chosen after verification: **CICIoMT2024** (same UNB extractor, 44
columns shared; WiFi/MQTT part only) and **Edge-IIoTset** (agriculture sensors over MQTT; 14
attacks; per-packet Wireshark rows; a documented `"0"`/`"0.0"` provenance artefact, arXiv
2608.15761). Everything below is written from the published documentation and tested on
synthetic frames shaped like it; **it is not validated on the real files**, and
`docs/plans/phase9-multi-dataset.md` §3 says what to reconcile first.

- `data/datasets.py` (new): `REGISTRY` of `DatasetSpec`s; `CANONICAL_COLUMNS` (CICIoT2023's 39
  raw columns) and `SELECTED_COLUMNS` (Phase 2's 16); `harmonise_columns` (aliases, identifier
  drops, never invents a column — gaps reported as `missing_canonical`);
  `normalise_placeholders` / `placeholder_collisions` for the artefact.
- `data/taxonomy.py`: `CICIOMT2024_LEAF_TO_FAMILY` (19) and `EDGE_IIOTSET_LEAF_TO_FAMILY` (15)
  into the fixed C=8 families; `to_family` / `to_class_index` take an optional map, default
  unchanged. Judgement calls documented in the module.
- `data/packet_windows.py` (new): `aggregate_packet_windows` (windows never cross a label
  change or a source file; partial tail windows dropped; only supplied ingredients produce
  columns) and `edge_iiotset_packets` (derives length/flags/transport from the documented
  Wireshark columns; fails loudly on an unexpected layout).
- `data/characterize.py`: optional `parts`/`label_of` so a registered corpus is scanned with its
  own discovery and label derivation; new `placeholder_collisions` report field.
  `scripts/run_phase1.py --dataset <name> [--root PATH]` → `asa characterize --dataset …`.
  Verified on synthetic CICIoMT2024 and Edge-IIoTset trees: labels derived from filenames /
  the `Attack_type` column, and the artefact warning fires.
- `tests/test_datasets.py` (18 tests) and `tests/test_taxonomy.py` (error message generalised).
- **Reconciled against the CICIoMT2024 README (same day).** Its WiFi/MQTT feature table lists
  39 features whose names and definitions are the CICIoT2023 raw vocabulary verbatim — the
  "44 of 47" figure from the literature search referred to the Kaggle-era CICIoT2023 columns,
  not the raw distribution this project uses. Consequences: the guessed `Duration → Time_To_Live`
  alias was removed; `datasets.canonical_name` maps any spelling ("Header Length",
  "Time-To-Live", "Tot Sum") onto the canonical name, so the CSV header spelling no longer
  needs guessing; `discover_parts` gains `exclude_dirs` and CICIoMT2024 skips `profiling/`,
  `Bluetooth/` and `PCAP/`; the TCP/IP flood leaves are mapped under both the filename
  (`TCP_IP-DDoS-ICMP`) and README-chart (`DDoS-ICMP`) spellings. `packet_windows.py` now
  follows the README's definitions — Header_Length is a mean (was sum), Protocol Type the
  window mode (was mean), and `Rate` (packets/s) is produced. Tests: 22 in `test_datasets.py`,
  including one asserting the README table maps onto `CANONICAL_COLUMNS` exactly.

Repo bookkeeping: `CLAUDE.md` phase list (8 status, 9 added), commands, invariants (sealed
weights; placeholder collisions; harmonisation never invents), and the dataset scope line.
Full gate set green: ruff, ruff-format, mypy (50 files), vulture, 390 tests, pre-commit.

## 2026-09-18

### Architecture redirect: Ascon now protects the weight exchange, and the paper follows the architecture

The user redefined the system and stated that the design paper is no longer the authority;
it will be amended to match. The new specification, recorded in
`docs/plans/phase8-hardware-federation.md`: per-farm ESP32 sensors feed a **local GRU** on a
Raspberry Pi; local GRUs are clients of a **master (federated) GRU** on a laptop; the
**model weights are Ascon-AEAD128 encrypted in both directions** (local → master update,
master → local global state) because that exchange crosses the untrusted IoT network; the
**local GRU takes the benign/malicious decision** with the weights it receives back and is the
only party that talks to the cloud. The plan fixes per-client per-direction keys, an
authenticated-data layout `⟨client_id, round, direction, schema_version⟩`, a length-prefixed TCP
frame protocol, the module/test list, and the Pi feasibility constraints (Python ≥ 3.12 rules
out Pi OS Bookworm; cap per-client sequences for the hardware run).

Changed in this piece of work: `CLAUDE.md` golden rule 1 (architecture overrides the paper),
the phase list (Phase 8 added), the invariants (weights cross the network only Ascon-sealed),
and the out-of-scope line (physical hardware is in scope as the Phase 8 demo nodes). No source,
test, or config was touched; the phase-status table gains a "Planned, not started" Phase 8 row.
Assumption still to confirm with the user, written into the plan: ESP32 "attacks" are
**scenario-triggered replay** of held-out CICIoT2023 records (option A) rather than live traffic.

**2026-09-19 follow-up:** the user confirmed Ascon on the weight channel is definite (both
directions, so a hacker manipulating weights is detected and rejected), and asked which cloud-hop
option is better. Decision: the Pi → cloud hop uses **plain TLS** and Ascon is *not* layered
under it — one cipher per hop, matching real-world practice, keeping the project's security claim
on the weight channel alone. The cloud receives the benign reading, never a verdict; the
benign/malicious routing and the path-disjointness test are unchanged. The plan doc was updated
to match (§1, §3, §4, §5, §8).

## 2026-09-16

### Added a client-count (K) scalability analysis -- flagged as outside the paper's scope

`scripts/analyze_k_threshold.py` and `scripts/run_k_sweep.py` ask how many edge gateways can
join one federation before the detector degrades. **This is not one of the paper's ablations**,
and is recorded as such per Golden Rule 1: K is fixed at 3 by assumption A1 ("Three edge nodes
are simulated") and objective O3, and `configs/base.py` defines `alpha_sweep`, `e_sweep`,
`w_sweep` and `f_sweep` but no `k_sweep`. It was added at the user's request, after the
discrepancy was surfaced rather than resolved silently. The phase-status table above is
deliberately unchanged: this advances no phase and satisfies no paper exit criterion.

The analytical half trains nothing. Reading the per-class block counts back out of
`artifacts/manifest_phase4_default.json` (4,854 train blocks; rarest class **BruteForce at 42**),
it establishes three results:

- **Pigeonhole, exact at any α.** Blocks are indivisible, so a class with `B_c` blocks reaches at
  most `min(K, B_c)` clients. Past K = 42, some gateway necessarily holds zero BruteForce traffic.
- **Dilution, exact as α → ∞.** `ω_c(K) = min(1, B_c/K)`, where ω_c is the share of the Eq. (21)
  aggregate contributed by clients that have actually seen class c — a quantity defined here, not
  in the paper. Checked against the Monte-Carlo by `--verify`.
- **Bandwidth, exact.** Eq. (22) gives 270,400·K bytes/round, reproducing the manifest's measured
  811,200 B at K = 3; over R = 20 rounds that equals the paper's own 276 MB pooling figure at
  **K = 51**.

At the project's α = 0.5 the Monte-Carlo puts ω(BruteForce) through the 0.5 floor at **K ≈ 48**,
so two independent constraints place the ceiling near 50 — and the binding quantity is a property
of the *data* (`B_min`), not of GRUs or of FedAvg.

A closed form for finite α was attempted and **rejected rather than shipped**: the Beta-marginal
approximation `1 − I_{1/B_c}(α+1, α(K−1))` overestimates ω by up to 0.80, because
`dirichlet_block_partition` hands each client a contiguous slice of the shuffled block list
(`cuts = cumsum(p) * len(blocks)`) and so couples consecutive clients instead of leaving their
shares independent. Finite-α figures are therefore measured by running the real partition
function 240 times per point, not derived. The rejected form is documented in the module
docstring so it is not re-attempted.

`run_k_sweep.py` runs the matching empirical sweep over K ∈ {3, 5, 10, 20, 30, 50} × 3 seeds,
**baseline 5 only**: the centralised upper bound is K-independent by construction and is quoted
from Phase 4 rather than retrained, and local-only would mean 150 GRUs at K = 50 for a bound the
sweep is not asking about. Everything else matches Phase 4 exactly so its K = 3 point reproduces
the published figure. It records ω from the true Eq. (21) sequence weights rather than the
block-count proxy, so the prediction is checked against the quantity it was predicting.

Outputs: `artifacts/k_threshold_analysis.json` (written), plus
`artifacts/k_sweep_results.json` and `artifacts/manifest_ksweep_default.json` from the sweep.

## 2026-09-15

### Phase 7 checkpoint regenerated on the fixed training path

`artifacts/federated_global_model.safetensors` (gitignored) and both Phase 7 manifests were
re-created with the fixed `train_federated_model.py` (previous entry). **Macro-F1 0.8507,
bit-for-bit identical to seed 0 of the compute-matched Phase 4 run** (`0.8507461612861715`;
`summarize_phase7.py` reports `reproduces Phase 4 seed 0: True`), which closes the "checkpoint
vintage" caveat in `results/phase7_end_to_end_integration.md`: the deployed model is now the
Phase 4 run's own seed-0 model, trained on its partition and its federated scaler. 18.6 min
from the Phase 4 cache. The end-to-end run on the new checkpoint gives the same routing counts
as before (100 messages, 55 classified, 0 benign / 55 malicious, 0 malicious-verdict payloads
at the cloud, 53/55 informal agreement); per-stage medians 1.2 / 3.9 / 112.0 / 2.4 µs.

### End-to-end verification of all seven phases; four reproducibility breaks fixed

Every phase was executed in order on this machine from the raw corpus (Phase 1 report
re-derived and byte-identical; Phases 2-4 from raw at a reduced budget, both gates closed;
Phase 6 benchmark at 96 B and 512 B; a checkpoint trained and run through Phase 7). The
committed full-budget manifests still close both gates. What broke on the way, and the fix:

- **`scripts/train_federated_model.py` did not train on Phase 4's partition or scaling**,
  despite its "IDENTICAL partition" comment. It re-cut blocks from the train labels (4,842
  blocks against the cache's 4,854, a different Dirichlet draw: client sizes 1164/1849/1829
  against Phase 4's 1166/1855/1833) and, when given the Phase 4 cache, fed the GRU **unscaled**
  features (the cache stores raw values; the script assumed pre-scaled ones -- a smoke run at
  R=2/E=1 scored 0.50 before the fix and 0.71 after, same seed, same data). It now calls
  `run_phase4.py`'s own `build_pipeline`, the new shared `block_strata`, and the federated
  scaler (Eqs. 23-24), and reproduces Phase 4's seed-0 sequence counts exactly
  (284,500 / 458,447 / 452,390). The deployed checkpoint and both Phase 7 manifests are
  regenerated from the fixed script in the entry that follows.
- **The same script hard-coded the superseded 0.8334 +/- 0.0026** from the inverted
  minimal-gate run as the Phase 4 reference and wrote it into every Phase 7 training manifest.
  It now reads the 3-seed mean +/- std from `--phase4-manifest`
  (default `artifacts/manifest_phase4_default.json`) and records which file it read.
- **`scripts/summarize_phase7.py` crashed** (`KeyError: 'federated_per_seed'`) against the
  current Phase 4 manifest: it still read the minimal-gate schema. Ported to `per_seed.federated`.
- **The `asa` CLI raised `NotImplementedError` for every phase** while its docstring promised
  one-binary reproducibility. It now dispatches to the driver under `scripts/` with arguments
  passed through (`asa federate --rounds 2`), and says plainly that Phases 2, 5 and 6 have no
  standalone driver (exit 2). `scripts/run_phase1.py` added so Phase 1's committed report has
  a script that produces it. `tests/test_cli.py` (3 tests) covers the dispatch.

### Merged two parallel Phase 4 implementations

Two Phase 4 drivers were written independently and collided on merge (7 conflicted paths).
Resolved by union rather than by picking a side; what each contributed:

- **`federated_global_gru` keeps the 4-tuple contract** `(metrics, macro_f1_per_round,
  measured_bytes_per_round, model)`. Returning the trained model is not optional —
  `scripts/train_federated_model.py` and Phase 7's runtime need an actual classifier to drive
  routing, and the single-metrics form could not supply one. It now delegates to
  `run_federation`, which additionally carries the Eq. (21) weights and the per-round byte
  series the Phase 4 manifest records, so both callers are served by one implementation.
- **A local-only client with zero sequences is reported, not filtered.** The competing revision
  returned `None` so such a client could be excluded from the mean. That was wrong in the
  direction the bracket exists to expose: Section III-I1 asks for "three local-only GRUs" as a
  report, and dropping the client that received nothing makes the lower bound look better than
  declining to federate actually is. It is now scored against a constant-benign prediction.
- **An explicit all-empty guard** in `run_federation`, raising "every client holds zero
  sequences" at the top rather than letting the failure surface from `weighted_fedavg` as
  "every client reported n_k = 0" — the symptom, not the cause.
- **FedProx (`fedprox_mu`) and the (run seed, client id, round) training seed coexist** in
  `FederatedClient`; only their docstrings conflicted, and both paragraphs are kept.
- **Section III-I2's client-to-global gap** is now computed by `run_phase4.py` and recorded per
  seed. It had existed only in the other driver's post-hoc summariser; on the merged run it is
  positive for every client (+0.1023 / +0.1565 / +0.0721), i.e. federation helped all three,
  which is a stronger statement than the mean alone.
- **`scripts/summarize_phase4.py` was ported** to the surviving manifest schema, keeping the G4
  questions it asks verbatim.
- **Both experiments are kept.** `artifacts/manifest_phase4_default.json` is the compute-matched
  run (161 min); `artifacts/manifest_phase4_minimal_gate.json` is the earlier 441-min run.

Two things the merge exposed that are worth stating plainly:

- **The minimal-gate run has the inverted bracket this repository has already fixed once.** It
  reports federated 0.8334 against a centralised *reference* of 0.8297 borrowed from Phase 3 at
  10 epochs, and a local-only baseline capped at 10 epochs, while federation ran R*E = 60 local
  passes. "Beating the centralised reference on every seed" is a statement about the training
  budget, not the method. Its manifest is retained as a record; its conclusion is not.
- **Git's auto-merge silently dropped test coverage.** `tests/test_federated.py` merged without
  conflict markers to 23 tests, from sides holding 26 and 25 — every test either side had added
  was gone, including both FedProx tests and all three training-seed tests. Rebuilt as the
  union (28). A clean merge is not evidence of a correct one.

Suite after the merge: **344 passed**, all gates green, `check_phase4_gate.py` exit 0.


### Phase 4 gate CLOSED — the federated simulation, run on the real corpus

`scripts/run_phase4.py` ran end-to-end on the official UNB CICIoT2023 distribution and
`scripts/check_phase4_gate.py` exits 0 on the resulting manifest
(`artifacts/manifest_phase4_default.json`). Every number below is mean ± std over seeds
{0, 1, 2} (III-I4); nothing here comes from synthetic data.

Corpus as prepared by Phases 1-2: 46,776,700 raw records -> capped subsample -> dedup -> block
split -> four-stage selection at F=16, giving **1,237,911 train / 311,565 test rows** across
4,854 train blocks, 1,222,009 pooled training sequences at W=16. Wall clock 160.8 min.

| baseline (III-I1) | macro-F1 | balanced acc. | MCC | accuracy |
| --- | --- | --- | --- | --- |
| 4. local-only GRUs (lower bound) | 0.7205 ± 0.0750 | — | — | — |
| 5. **federated global GRU** | **0.8308 ± 0.0150** | 0.8404 ± 0.0088 | 0.8845 ± 0.0174 | 0.9090 ± 0.0151 |
| 3. centralised GRU (upper bound) | 0.8543 ± 0.0040 | 0.8612 ± 0.0035 | 0.9021 ± 0.0020 | 0.9235 ± 0.0020 |

**The G4 answer:** the federated model sits inside its bracket and recovers **82.4 %** of the
macro-F1 that local-only training leaves on the table, for **815,136 measured bytes per round**
(Eq. 22 predicts 811,200; the 0.5 % excess is safetensors framing). Local-only is also by far the
least stable baseline (± 0.0750 against the federated ± 0.0150), because its score depends
entirely on which classes its client happened to be dealt: at seed 1, where two clients were
missing classes, local-only fell to 0.6145 while the federated model still reached 0.8145.

**Read the FPR before reading anything else.** Under the Eq. (5) binary projection the federated
model has a false-positive rate of **0.2975 ± 0.0512** (centralised 0.2767 ± 0.0283) at ~98.8 %
attack recall — roughly **three in ten benign flows raise an alarm**. Section III-I2 makes FPR
first-class precisely because this is the number that gets a detector switched off in
production, and no amount of 0.92 accuracy compensates for it. Accuracy (0.9090) sits below the
0.98 near-ceiling threshold, so the III-I5 note correctly does not fire — this run is not in
the dataset-artifact regime, and the weak classes are real: BruteForce 0.6339, WebBased 0.6522
and Benign 0.7148 per-class F1, against Mirai 0.9849 and Spoofing 0.9673.
  - These FPR figures are **derived from the confusion matrices in the committed manifest**, not
    separately measured: FPR is a pure function of the benign row. `run_phase4.py` now records
    `false_positive_rate` per seed and carries it in the headline set, so the next run stores it
    directly; this manifest predates that and was not re-run for a derived quantity.

**Eqs. (23-24) verified on real data:** the federated scaler, built only from per-client
count/mean/M2, matched a pooled fit to a maximum relative gap of **5.42e-12** across all three
seeds -- the III-F4 claim holding on 1.2M real rows, not only in a unit test.

### Fixed: the bracket was measuring the training budget, not the method

The first real seed put the federated model (0.8507) **above** its own centralised upper bound
(0.8293), which reads as "federation beats pooling" and is nothing of the sort. Federation makes
R*E = 20*3 = 60 local passes; baseline 4 was already matched to that, but baseline 3 was left at
the Phase 3 default of 10 epochs — a 6x training advantage handed to the method under test. With
the budget equalised the centralised baseline rose to 0.8584 on that seed and the ordering came
right. `--central-epochs` now defaults to `R*E`, the manifest records an `epoch_budget` block,
and `check_phase4_gate.py` gained an item that **fails** if the three budgets ever diverge again.
The run was stopped and restarted rather than allowed to finish and publish the inverted result.

### Changed: the centralised baseline is windowed over the whole training split

It previously trained on the concatenated client windows. "Upper bound attainable by pooling"
means a trainer that never sees the partition, so its windows must not be fragmented at
partition boundaries the way a client's are; concatenating handed the upper bound the federated
setting's handicap. It also matches how Phase 3 built this baseline, which is what makes the two
phases comparable — and that comparability is now evidence: at Phase 3's own 10-epoch budget
this pipeline reproduced Phase 3's centralised macro-F1 to **0.8293 against 0.8297 ± 0.0013**,
an independent end-to-end check that Phases 1-3 replay faithfully on this machine. Building the
pooled tensor before the per-client ones and freeing it also removed a ~1.2 GB duplicate from
peak memory.

## 2026-09-14

### Added: Phase 4 gate checker, and a real fix to the local training seed

- **`scripts/check_phase4_gate.py`** -- the machine-checkable answer to "is Phase 4 done?",
  mirroring `check_phase3_gate.py` so closing the phase is not a judgement call. It reads a
  Phase 4 manifest and checks: >= 3 seeds; baselines 3-5 reported with mean +/- std; accuracy
  never reported alone (III-I2); `aggregation == "weighted"` and the applied weights equal the
  per-client **sequence** counts rather than the row counts (Eq. 21); the global model reached
  all K clients; per-client class histograms published with a shared vocabulary so an absent
  class shows as 0 (III-F1 / G3); the federated-vs-pooled scaler gap within floating-point
  tolerance (Eqs. 23-24); communication cost measured against Eq. (22); the G4 bracket; and a
  manifest carrying config, versions and a real git commit (III-I4).
  - Like the Phase 3 checker it deliberately does **not** check that federation performed well.
    A federated model at the bottom of its bracket, properly measured, closes the gate.
  - **Verified to fail, not just to pass.** A gate that cannot reject is worse than none, so it
    was run against eight deliberately broken manifests -- unweighted aggregation, weights
    swapped to row counts, two seeds, a corrupted scaler gap, dropped histograms, a missing
    bracket, an absent git commit, and a client the global model never reached. All eight exit
    non-zero with the relevant item marked FAIL.

- **`FederatedClient` now seeds local training from (run seed, client id, round)** via a
  `SeedSequence`, replacing the bare `client_id`. The old seeding had two consequences that
  never surface as a crash, only as distorted numbers: within a run every round re-seeded
  identically, so a client replayed the *same* batch permutation in round 20 as in round 1 and
  the shuffle stopped being a shuffle after the first round; and across runs local training was
  independent of the experiment seed, so the >= 3-seed spread of Section III-I4 sampled only the
  initial parameters and the Dirichlet draw and reported a tighter std than the method actually
  has. This was raised twice as an open question before the real run rather than discovered
  after it. `training_seed()` is exposed so a test can pin the three inputs apart, and three
  tests in `test_federated.py` assert rounds differ, the run seed matters, clients differ from
  each other, the derivation stays reproducible, and the round counter actually advances.
  `run_federation` passes the run seed down to each client. Suite: **250 passed, 1 skipped**.

### Still blocking Phase 4 (the phase is NOT closed)

The federated experiment has still not been run on CICIoT2023, because the corpus is not on this
machine: `data/ciciot2023_raw/` does not exist, and there is no prepared cache. Everything above
was verified on synthetic arrays, which establishes that the runner and the gate behave
correctly and **nothing else** -- no macro-F1, bracket position, or convergence curve from a
synthetic run is a finding, and the synthetic manifest produced while testing the gate was
deleted rather than committed. Phase 4 closes when `scripts/run_phase4.py` has been run against
the real corpus and `scripts/check_phase4_gate.py` exits 0 on the resulting manifest.

### Changed: toolchain pinned to one set of versions; pre-commit hook actually installed

`pre-commit install` had never been run in this checkout, and the config had drifted from the
`dev` extra badly enough that installing it would have baked a contradiction into every commit.

- **Hook revs realigned with `pyproject.toml`'s `dev` extra:** ruff `v0.6.9` -> `v0.16.7`,
  vulture `v2.13` -> `v2.16`, mypy `v1.11.2` -> `v2.3.1`. pre-commit builds each hook its own
  isolated environment at the pinned rev, so a drifted pin means the hook and a local
  `ruff check .` are *different programs* -- one can pass while the other fails. vulture is the
  sharpest case: `pyproject.toml` pins it with an exact `==2.16` while the hook asked for 2.13,
  so the config contradicted itself. A comment at the top of the config now says to keep them
  in step.
- **`ruff` hook id -> `ruff-check`**; the bare `ruff` id is a deprecated alias in ruff-pre-commit
  >= 0.12 and emitted a warning on every run.
- **Python floor raised to 3.12** (`requires-python`, ruff `target-version`, mypy
  `python_version`). This is forced by the dependencies rather than chosen: numpy (>= 2.4) and
  scipy (>= 1.16) both declare `requires-python >= 3.12`, so `>= 3.11` was already a false
  claim -- the pinned stack cannot install on 3.11. It also un-blocked mypy, which under a 3.11
  target refuses to parse numpy's bundled stubs (they use PEP 695 `type` statements) and so
  failed before reaching any of our code.
- **`features/selection.py` Stage 2 now indexes the Spearman matrix as a float array**
  (`.to_numpy(dtype=np.float64)` plus a column->position map) instead of `spearman.at[row, col]`.
  Under the current pandas-stubs, `.at` is declared as a union spanning `str`/`bytes`/`datetime`,
  which cannot be compared against a float threshold; `float(...)` does not fix it either, since
  the union includes members `float()` rejects. A correlation is a float, so the array is the
  honest type. **Behaviour is unchanged** -- all 18 feature-selection tests still pass. This was
  a pre-existing failure surfaced, not caused, by the pin alignment.
- **`pre-commit install` run**; `pre-commit run --all-files` is green on every hook.

### Changed: setup docs now cover both platforms, and describe a venv that exists

`CLAUDE.md`'s Commands section documented `./.venv/Scripts/ruff.exe` -- Windows paths -- while
also claiming the scientific stack was "already installed in the system Python 3.11" and that
`.venv` was created with `--system-site-packages`. On a POSIX checkout none of those commands
run, and the system-Python claim is no longer true anywhere.

- `CLAUDE.md` and `README.md` now give **both** macOS/Linux (`.venv/bin/`) and Windows
  (`.venv\Scripts\*.exe`) invocations for setup and for every gate, state the 3.12 floor and
  why it exists, and drop the stale `--system-site-packages` / system-Python framing: everything
  installs into a project-local `.venv`.
- Documented the two things that actually bite: the pytest hook is `language: system` so the
  venv must be **activated** (not just addressed by path) when committing, and hook revs must
  track the `dev` extra.
- Dropped README's "add `,crypto` once the Ascon backend is chosen" -- there is no `crypto`
  extra; the backend is vendored.

### Added: Phase 4 baselines 4-5 and the federated experiment runner

- **Baselines 4 and 5 of Section III-I1 implemented** in `eval/baselines.py`, the last two
  `NotImplementedError` stubs Phase 4 owed. `local_only_grus` trains one GRU per client on that
  client's partition alone; `federated_global_gru` runs R rounds of Algorithm 1 and scores the
  resulting global model. Together with baseline 3 they form the bracket that answers gap G4.
  - **Signatures widened to take an explicit test set**, exactly as baselines 1-3 were: the
    scaffold's `(client_seqs, client_y, seed)` cannot express a train/test split and so could
    not return test metrics at all. Flagged here as a scaffold correction, not worked around.
  - **Every baseline is scored on the shared global test set** (Section III-B3), local-only
    clients included. Scoring a local model on its own partition's held-out slice would ask it
    an easier, different question -- its partition is class-skewed by the Eq. (20) draw -- and
    would make baselines 3, 4 and 5 three unrelated numbers instead of a bracket.
  - **A client with zero sequences yields `None`, not a zero-filled metric bundle.** Under
    alpha = 0.1 a client can legitimately receive no blocks (see `federated/partition.py`) and
    has no detector to report. A bundle of zeros would silently drag a reported mean down as
    though the client had trained and failed. This widens baseline 4's return type to
    `list[MulticlassMetrics | None]`, deliberately.
  - **Per-client seeds come from a `SeedSequence` spawned off the run seed**, so client 0 at
    seed 1 is not the same run as client 1 at seed 0.
  - `run_federation` returns a `FederatedRun` carrying the pieces the manifest needs beyond the
    headline metric: the Eq. (21) weights actually applied, the Eq. (22) bytes actually sent per
    round, and (optionally) the per-round convergence curve. `federated_global_gru` is a thin
    wrapper over it so the Section III-I1 baseline contract stays a single metric bundle.

- **`scripts/run_phase4.py`** -- the federated experiment, mirroring `run_phase3_complete.py`.
  Runs baselines 3-5 over >= 3 seeds at a given alpha, publishes the per-client class histograms
  (Section III-F1 / gap G3) including zero counts, reports the bracket, and writes a manifest.
  - **The global scaler is built federatedly and is now actually exercised** (Eqs. 23-24).
    `build_pipeline` deliberately does **not** standardise: clients emit count/mean/M2 only, the
    server combines them, and the scaled arrays are derived from that result. An earlier draft
    scaled with a pooled fit in the pipeline and computed the federated statistics beside it,
    which made the III-F4 path dead code; that was wrong and is fixed. The run prints the
    federated-vs-pooled gap as a live check (4e-15 on the synthetic smoke run).
  - Carries its **own cache format** (`--save-cache`/`--cache`) because Phase 4 needs the
    per-row block ids that the Phase 3 cache does not store.

- **Tests:** `test_phase4_baselines_remain_gated` -- which asserted the two stubs still raised --
  is removed, being obsolete the moment they were implemented, and replaced by seven tests
  covering the shared-test-set property, the empty-client `None`, seed independence, learning,
  the published Eq. (21)/Eq. (22) figures, and malformed-partition rejection. Suite: **247
  passed, 1 skipped** (the remaining skip is Phase 6 routing).

### Not done, and not to be reported as done

- **The federated experiment has not been run on CICIoT2023.** The runner was verified end-to-end
  on synthetic arrays only; every macro-F1 it printed there is noise from generated data. No
  Phase 4 number is a finding until the run happens on the real corpus, and the synthetic-data
  manifest that smoke run produced was deleted rather than committed.
- **Observation for the team, not silently changed:** `FederatedClient.local_train` seeds
  `train_module` with `self.client_id`, so a client's batch shuffling is identical across
  experiment seeds. Run-to-run variance therefore comes only from the initial global parameters
  and the Dirichlet partition, which under-samples the spread that III-I4's mean +/- std is
  meant to report. Left as-is because it is Phase 4 infrastructure someone else wrote and its
  tests pass; worth a decision before the real run.

### Added

- **Phase 3 completed: scaling, training loop, metrics, reporting guard and manifest.**
  New `data/scaling.py`; implemented `model/train.py`, `eval/metrics.py`, `eval/report.py` and
  `eval/manifest.py`; new experiment driver `scripts/run_phase3.py`. New `tests/test_scaling.py`
  (8), `tests/test_metrics.py` (9), `tests/test_training.py` (15), `tests/test_manifest.py` (4).
  - **Pooled scaler kept out of `federated/`.** The Phase 3 scaler lives in `data/scaling.py`
    rather than in Phase 4's `federated/scaler_stats.py`, so Phase 4's module stays closed until
    its gate opens. Statistics are stored as `(count, mean, M2)` — the form Chan's parallel
    formula (Eqs. 23-24) combines — specifically so `tests/test_scaler_equivalence.py` has a
    real pooled fit to compare the federated combination against when Phase 4 starts.
    Zero-variance features are divided by 1.0, not 0.0.
  - **Non-finite policy for training data, decided and documented.** The selected F=16 columns
    carry **47 NaN cells in `Variance`** (0.004% of training rows, genuine nulls in the raw
    CSVs). `fit_scaler` *refuses* non-finite input rather than imputing, and the driver drops
    the affected rows. Dropping leaves a positional gap, which `contiguity_segments` turns into
    a segment break — so no window spans the hole, exactly as none may span a held-out block.
    Measured: 47 rows dropped from train, 8 from test.
  - **Class-weighted CE (Eq. 18) with a guard the paper does not specify:** a class absent from
    the training split gets weight **0**, not infinity. Eq. (18) divides by n_c, undefined at
    n_c = 0; an infinite weight would let one stray prediction dominate the loss. The absence
    stays visible in the per-class F1 instead.
  - **Metrics honour the "never accuracy alone" invariant, and the macro average includes
    classes absent from the split** (F1 = 0 rather than dropping them from the denominator,
    which would flatter a detector precisely on the rare families). PR-AUC returns NaN — not a
    fabricated 0.0 or 1.0 — when only one class is present. A regression test asserts a
    majority-class detector scoring 0.96 accuracy is correctly punished to macro-F1 < 0.35.
  - **The Section III-I5 near-ceiling guard is live**: accuracy at or above the configured
    threshold automatically emits the note directing the reader to macro-F1 and FPR.
  - **Manifest writing implemented**, capturing seeds, full config snapshot, library versions,
    platform, git commit **and whether the tree was dirty** — a dirty tree means the commit does
    not fully identify the code that produced the number.
  - **PHASE 3 EXIT CRITERION MET — the centralised GRU trained and evaluated on the real
    corpus**, 3 seeds (0/1/2), W=16, F=16, 3 epochs, 1,222,009 training and 296,750 test
    sequences, ~11 min/seed on CPU. Manifest at `artifacts/manifest_phase3_default.json`
    (gitignored). Reported as mean ± std over seeds, never a single run:

    | Metric | Result |
    | --- | --- |
    | macro-F1 | **0.7866 ± 0.0199** |
    | macro precision / recall | 0.7575 ± 0.0129 / 0.8660 ± 0.0084 |
    | balanced accuracy | 0.8660 ± 0.0084 |
    | MCC | 0.8532 ± 0.0097 |
    | binary PR-AUC | 0.9993 ± 0.0000 |
    | **binary FPR** | **0.2573 ± 0.0612** |
    | accuracy (alongside only) | 0.8818 ± 0.0080 |

    Per-class F1: Mirai 0.999, Spoofing 0.945, DDoS 0.927, DoS 0.847, Reconnaissance 0.832,
    Benign 0.703, **WebBased 0.529 ± 0.063, BruteForce 0.512 ± 0.076**.
  - **Three findings that the evaluation protocol exists to surface, recorded rather than
    smoothed over:**
    1. **Accuracy is 0.88, not the 98-99% the CICIoT2023 literature routinely reports** — so the
       Section III-I5 near-ceiling note did *not* fire. This is the expected consequence of the
       leakage controls, not underperformance: R3's dedup-before-split removed 25.6M duplicate
       rows corpus-wide (54.8%) and block-level splitting keeps neighbouring records on one side
       of the split. Published near-ceiling figures on this benchmark are the number you get
       *without* those controls. The gap is evidence the controls bind.
    2. **The binary false-positive rate, 0.257, is far too high to deploy.** Roughly a quarter of
       benign windows are flagged as attack, which is precisely the failure mode Section III-I2
       makes FPR a first-class metric to expose ("a high false-alarm rate is what gets a detector
       switched off"). Binary PR-AUC of 0.9993 looks excellent and hides this completely — the
       threshold, not the ranking, is what is wrong. Tuning the operating point is future work.
    3. **The model is under-trained at 3 epochs.** Weighted CE was still falling when training
       stopped (0.575 → 0.377 → 0.311 per epoch, consistently across all three seeds). The
       reported numbers are therefore a floor, not the architecture's ceiling, and the two rare
       families (BruteForce, WebBased) also carry the largest seed-to-seed spread (±0.076,
       ±0.063). A longer run is expected to improve all three; the exit criterion is that the
       protocol is *reported*, which it is, not that the detector is good.
  - **Runner buffering fixed.** The first full run redirected stdout to a log file, where Python
    block-buffers a non-TTY stream — the log stayed empty for the entire 21-minute run, making
    progress unobservable. `scripts/run_phase3.py` now line-buffers its own output.

- **Phase 3 opened (user-authorised): taxonomy, windowing and the GRU detector implemented.**
  New `data/taxonomy.py`, implemented `sequences/windowing.py` and `model/gru.py`. New
  `tests/test_taxonomy.py` (10), `tests/test_windowing.py` (21), `tests/test_gru_detector.py`
  (9); `tests/test_model_param_count.py`'s Phase-3 skip is removed and three cases added.
  Pytest now 151 passed / 3 skipped (up from 102 passed / 4 skipped). Phase 3 is **not**
  complete: the training loop, feature scaling and the Section III-I evaluation protocol remain,
  so its exit criterion is unmet and Phase 4 stays closed.
  - **Taxonomy discrepancy reconciled (flagged per Golden Rule 1, confirmed with the user).**
    The paper states the class taxonomy twice and the statements disagree: Section III-B3's prose
    says "benign plus seven attack families" (C = 8), `ModelConfig` hard-validates
    `n_classes == 8` and Eq. (19)'s 33,800 parameters assume C = 8 — but **Table I lists only
    six family rows**, merging DDoS and DoS into one `DDoS / DoS` row. Splitting that merged row
    is the only reading satisfying prose, validator and Eq. (19) at once; the alternative (follow
    Table I literally at C = 7) was put to the user and rejected. `data/taxonomy.py` is now the
    single place the mapping lives. Two placements inside it are judgement calls following the
    canonical CICIoT2023 grouping, both recorded: `VulnerabilityScan` → Reconnaissance and
    `Backdoor_Malware` → WebBased. **Verified against the real dataset**: all 34 on-disk labels
    map, none unmapped, no mapped label absent from disk.
  - **Benign is fixed at class index 0**, so Eq. (5)'s binary projection is exactly `y > 0` and
    never needs a lookup. Unknown labels raise rather than falling back to a default class,
    which would otherwise corrupt the per-class metrics of Section III-I.
  - **Windowing implemented** with runs defined as maximal stretches sharing *both* label and
    source id, sequences emitted in input order (never shuffled — Section III-D reserves
    shuffling for complete sequences at batch time), labels taken from the final record
    (y_i = y_{i+W-1}), and Eq. (12) checked directly against `count_sequences`.
  - **Correctness bug found and fixed while composing the pipeline: windows could span a
    held-out block.** Splitting operates on whole blocks, so filtering the deduplicated frame
    down to the train blocks leaves rows that were never neighbours sitting adjacent — block 5
    and block 7 become adjacent when block 6 goes to test. Windowing over that filtered frame
    fabricates an adjacency the capture never contained, which the positional-ordering
    assumption of Section III-D does not license. Measured on the real training split: **977
    such discontinuities, which would have produced 14,508 bogus sequences (1.2% of the training
    set)** built on adjacency that never existed. Fixed by the new
    `windowing.contiguity_segments(source_ids, row_positions)`, which breaks a segment wherever
    the source file changes *or* the original row positions stop being consecutive; callers pass
    its output as `build_windows`' `source_ids`. Regression tests cover both break causes.
  - **`build_latency_eval_windows` gained a keyword-only `benign_label`** (a scaffold signature
    extension): the module needs to know which class counts as benign to find benign→attack
    transitions, and taking it as a parameter keeps `sequences/` independent of
    `data/taxonomy.py`'s indices, working on leaf labels or family indices alike.
  - **GRU detector implemented** as GRU → LayerNorm(h_W) → Linear, reading the final hidden
    state (consistent with y_i = y_{i+W-1}). LayerNorm never BatchNorm, asserted by a test that
    walks the module tree. The default sizing reproduces **exactly 33,800 parameters**, matching
    Eq. (19) term for term (GRU 32,832 + LayerNorm 192 + head 776), and the formula/built-model
    equality is now checked at four further sizings rather than only the reference point.
    `torch` moved from a `TYPE_CHECKING` guard to a real module-scope import.
  - **Tooling:** a narrowly-scoped `disallow_subclassing_any = false` override for
    `ascon_smart_agri.model.*` in `pyproject.toml`. `torch` is deliberately not a pre-commit
    mypy `additional_dependency` (it would add a multi-hundred-megabyte download to the hook
    environment for a stubs-only benefit), so `nn.Module` resolves to `Any` there and strict
    mode's `disallow_subclassing_any` fired on the detector, while the local `mypy src` run —
    which does see torch — type-checks the module properly. Both environments are now green.
  - **Full chain verified end-to-end on real data**: raw CSVs → subsample → dedup → split →
    four-stage selection → taxonomy → windowing → GRU forward pass. 1,237,958 train rows → 34
    leaves mapped to 8 classes → F=16 of F0=31 → **1,222,745 sequences** of shape (16, 16) over
    1,015 runs → (64, 8) logits from a 33,800-parameter model. At family level the imbalance
    ratio is **44.7**, not the leaf-level IR ≈ 5751 the paper quotes — which is exactly the
    motivation Section III-B3 gives for reporting at family granularity. Benign is 4.5% of
    training rows.

### Phase 7 complete: real model trained, saved, and run through the real pipeline

Closes the last item from the gap audit: `scripts/train_federated_model.py` trains and saves
(via `model/checkpoint.py`, new this entry) one federated global model checkpoint under the
IDENTICAL Phase 4 configuration and data partition -- same seed, same α=0.5, R=20, E=3,
weighted FedAvg. Scope decision (flagged): seed 0 only, not a fresh 3-seed run, since Phase 4
already established the statistical claim and Phase 7 needs a real deployment artifact, not a
second validation. 54.8 minutes (faster than Phase 4's contended 78.6-minute seed-0 run; no
contention this time). **Result: macro-F1 0.8338 -- bit-for-bit identical to Phase 4's seed-0
run** (verified against Phase 4's own manifest, not a hand-typed constant -- see the bug note
below), confirming the training path is genuinely deterministic end to end.

`scripts/run_phase7.py` is the first script in this project to execute the paper's own
sentence for real: *"telemetry arrives, is classified using the current global model, and is
routed according to Equation (5)."* One message at a time, in order:
`simulate_stream` → `FeatureProvenanceAdapter` (G6, held-out only) → `DeviceWindowBuffer` (new
this entry, `sequences/streaming.py`) → the real saved model → Eq. (5)'s `y > 0` projection →
`VerdictRouter` → real `AsconAEAD128`/`MockCloudReceiver` or `AlertSink`. Validated with a real
smoke run (a tiny 2-round checkpoint on sliced data, 80 messages) before the expensive training
finished, then run for real against the trained checkpoint.

**Real end-to-end results** (100 simulated messages, 3 devices, W=16): 45 still buffering
(correct -- no device had accumulated a full window yet, no padding was fabricated), 55
classified (0 benign, 55 malicious). Cloud received exactly 0, matching 0 benign verdicts
exactly; alert count exactly 55, matching every malicious verdict; **zero malicious-verdict
messages reached the cloud** -- G1 holds through the fully assembled runtime loop, not merely
in the isolated unit tests. Per-stage latency (median): provenance 3.6us, window 10.5us,
inference 273.7us, routing 4.1us -- inference dominates, as expected for a GRU forward pass
versus dict lookups and byte serialisation.

- **A real, verified finding, not a bug: this run's sample drew 0 true-benign records among
  its 55 classified windows' final records that the model called benign** (53/55 = 96.4%
  informal agreement with true_label -- NOT a formal evaluation, see the module's own
  disclaimer). Checked rather than assumed: the held-out pool's true benign rate is 4.52%, and
  the seeded draw for stream positions 45-99 (the ones that got classified, given 45 messages
  went to filling buffers first) landed exactly 2 true-benign records -- almost exactly the
  4.52% x 55 ≈ 2.5 expected by chance. Both of those 2 were (informally) misclassified as
  malicious, a small-sample (n=2) echo of the already-documented binary FPR limitation
  (0.257, measured formally in Phase 3) -- not new evidence, and far too small a sample to be
  one. Phase 6's own integration test already proved benign routing works correctly with real
  crypto (8 of 200 messages there were genuinely benign and all 8 round-tripped correctly); this
  run simply didn't draw one in its classified range.
- **A real bug in the verification script itself, caught and fixed before it shipped a wrong
  answer.** `scripts/summarize_phase7.py` first compared the checkpoint's macro-F1 against a
  hand-typed `0.8338` (the value as printed, truncated to 4 decimals) with a `1e-6` tolerance,
  and reported **"reproduces Phase 4 seed 0 exactly: False"** for a run that was in fact
  bit-for-bit identical -- the true value is `0.8338014523294024`, which differs from the
  truncated constant by `1.45e-6`, just over the tolerance. Fixed to read Phase 4's actual
  recorded seed-0 value from its own manifest and compare for exact equality, rather than
  trusting a copied-in digit string to stay in sync with the real number. Found by checking the
  comparison's own inputs rather than accepting "False" at face value.
- New `scripts/summarize_phase7.py` (distilling both `manifest_phase7_train_default.json` and
  `manifest_phase7_e2e_default.json` into `artifacts/phase7_results.json`, same relationship as
  Phase 4's summarizer). The model checkpoint itself
  (`artifacts/federated_global_model.safetensors`, 136 KB) stays gitignored as bulk output,
  consistent with the existing `*.safetensors` policy; the manifests and results file, being
  JSON provenance, are tracked.

### Phase 7 opened: three gaps closed that don't require retraining

Audited Section III-I2 against the codebase before writing any Phase 7 code (the Phase 3
lesson, applied proactively this time rather than after declaring something complete): grepped
for `client_to_global_gap`, encrypt/decrypt latency measurement, and any persisted model
weights. Found the first two never implemented and the third genuinely absent -- no training
run in this project has ever saved a model to disk. Closed the two that don't need retraining
now; the third (a real classifier driving the runtime pipeline) needs a fresh federated run and
is scoped separately below.

- **`sequences/streaming.py`, closing the gap `telemetry/provenance.py` explicitly deferred
  here.** `DeviceWindowBuffer` assembles `(W, F)` windows per device from single feature
  vectors arriving one call at a time -- the streaming counterpart to
  `sequences/windowing.py`'s offline, whole-batch construction. Rolling per DEVICE (not a
  single shared buffer, which would splice unrelated devices' readings together); a window is
  emitted the instant a device's buffer first reaches `W` and then slides by one on every
  subsequent push, consistent with `y_i = y_{i+W-1}`; nothing is ever padded to fake a window
  before real data fills it, the same objection that already rules out synthetic oversampling
  and fabricated network features. New `tests/test_streaming_windows.py` (12).
- **`eval/report.py`'s `client_to_global_gap`**, defined as `federated_macro_f1 -
  local_only_macro_f1` per client, both scored on the SAME shared test set (flagged, Golden
  Rule 1: the paper names this metric in Section III-I2 but never defines its sign or exact
  form). Positive means federation helps that client; negative is the G4/R4 finding the paper
  explicitly permits and asks to be reported honestly. **Retroactively applied to Phase 4's own
  results** -- `scripts/summarize_phase4.py` now computes it from the ALREADY-SAVED
  `manifest_phase4_default.json` (no retraining needed) and `artifacts/phase4_results.json` was
  regenerated: every client gains from federation, +0.063 to +0.100 macro-F1. New tests in
  `tests/test_training.py` (4).
- **`eval/crypto_benchmark.py`**, measuring the already-implemented, KAT-verified
  `AsconAEAD128` facade -- no new cryptography. A real bug in the benchmark itself, not the
  crypto, surfaced on its first run: `AsconAEAD128.encrypt()` returns only `ciphertext||tag`
  (payload + 16 bytes); Eq. (29)'s 32-byte wire-expansion figure additionally counts the nonce,
  which travels as a separate field in this design and isn't part of that return value. Fixed
  to add `len(nonce)` explicitly; the fix was found because the benchmark's own first run
  disagreed with the paper's worked 33%/6.3% examples, not assumed correct in advance. New
  `tests/test_crypto_benchmark.py` (7) checks the expansion exactly against those two worked
  examples. **Real measurements on this machine** (the vendored pure-Python reference backend,
  not an optimised implementation -- see the Phase 6 backend-selection entry):

  | Payload | Expansion | Encrypt (median / p95) | Decrypt (median / p95) |
  | --- | --- | --- | --- |
  | 96 B | 32 B (33.33%) | 390.8 us / 400.9 us | 391.9 us / 403.3 us |
  | 512 B | 32 B (6.25%) | 1300.5 us / 1327.8 us | 1302.1 us / 1330.3 us |

  Expansion matches the paper's stated 33%/6.3% exactly, since it is a structural property of
  the scheme (Eq. 29), not a measurement with noise -- latency is genuinely empirical and
  machine-dependent, reported as a distribution rather than a single number for that reason.

Pytest now **329 passed / 0 skipped** (up from 306/0).

### Phase 6 complete: alerting path, replay window, G1's behavioural half -- zero skips left

Implements `routing/alert_sink.py`, `routing/cloud_sink.py`, `routing/router.py`, and a new
`routing/replay_guard.py`, closing out Phase 6 (the crypto core was implemented ahead of its
gate back in Phase 6's first entry; this completes the routing half). New
`tests/test_replay_window.py` (9), `tests/test_cloud_sink.py` (8), `tests/test_router.py` (7),
`tests/test_alert_sink.py` (4); `test_path_disjointness.py`'s remaining skip activated with a
real behavioural test. **Pytest now 306 passed / 0 skipped** (up from 277/1) -- every test in
the suite is live for the first time in this project.

- **Replay-window policy confirmed with the user before any code was written**, per
  `docs/plans/phase6-replay-window.md`'s explicit instruction that this was not yet
  authorised. Chosen: per-`(edge_id, device_id)` scope, strict-monotonic acceptance
  (`counter > last_seen`), in-memory state -- the plan doc's own recommended sketch. Implemented
  as `routing/replay_guard.py`'s `ReplayGuard`, deliberately its own small object rather than
  folded into the crypto core, matching `AssociatedData`'s docstring which already stated
  replay state must not live there.
- **Verification order is fixed and tested as load-bearing**: `MockCloudReceiver.send_encrypted`
  decrypts/verifies (Eq. 26) FIRST, checks replay SECOND, never the reverse. A message with a
  tampered tag must never touch replay state, or a forged counter could poison the window for a
  legitimate later message -- `test_cloud_sink.py`'s
  `test_a_failed_verification_never_advances_replay_state` sends a corrupted message at
  counter=5 (rejected), then the genuine message at counter=5 (accepted), proving the counter
  was never consumed by the failed attempt.
- **`route()`'s scaffold signature (`edge_id: str` alone) could not build a valid Eq. (27) AD
  tuple** -- `device_id`, `counter`, `schema_version` were simply absent, so the benign path
  could not have encrypted correctly. Widened (flagged) to take a full `AssociatedData`, which
  already carries everything Eq. (27) needs, rather than adding three more scattered
  parameters. Same class of correction as the baseline-function widenings in Phases 3-4.
  `VerdictRouter` also gained an owned, cross-call-persistent `NonceRegistry` (injectable for
  testing) -- the scaffold's constructor had nowhere to draw nonces from at all.
  `MockCloudReceiver.__init__` gained a `keys: dict[str, bytes]` DEMO key store (explicitly
  labelled as such; production key management is named out of scope by the paper itself), since
  the receiver's own docstring already said it must "select the decryption key" and the
  scaffold gave it no way to.
- **G1's structural half is unchanged and still gating** (`AlertSink.__init__` takes no
  transport dependency; introspection confirms no instance attribute is ever a
  `CloudTransport`). The newly-activated behavioural half routes 5 real malicious-verdict
  messages through the real `VerdictRouter` and asserts the cloud receiver's
  `received_count` AND `rejected_count` both stay 0 -- not merely "nothing was accepted" but
  "nothing was even attempted", since the alert sink cannot reach the cloud transport to try.
- **Verified end-to-end on real data**, chaining Phase 5's telemetry/provenance layer into
  Phase 6's crypto/routing layer for the first time: 200 real simulated messages, each paired
  with a real held-out CICIoT2023 record via `FeatureProvenanceAdapter`, routed through the
  real `AsconAEAD128`/`VerdictRouter`/`MockCloudReceiver` stack using the record's true label as
  a stand-in verdict (never fed to a model -- a smoke-test substitute, since Phase 7 is where a
  real classifier gets wired in). Of 200 messages, 8 were benign and 192 malicious; the cloud
  received exactly 8, rejected 0, and every accepted payload decrypted to byte-identical
  original content; the alert sink recorded exactly 192, and the cloud saw zero of them.
- **Scope note:** replay persistence across a receiver restart, and the full III-G4 overhead
  measurement (wire expansion at multiple payload sizes) are not implemented -- the former is
  explicitly out of scope alongside production key management, the latter is a reporting task
  for Phase 7's end-to-end run rather than a Phase 6 gate item.

### Phase 5 complete: telemetry simulation + feature-provenance adapter (G6)

Implemented `telemetry/simulate.py` (`simulate_stream`) and `telemetry/provenance.py`
(`FeatureProvenanceAdapter`), plus a new `TelemetryConfig` in `configs/base.py`/`default.yaml`.
New `tests/test_telemetry_simulate.py` (11) and `tests/test_telemetry_provenance.py` (15) --
pytest now **277 passed / 1 skipped** (up from 251/1).

- **Two planes, kept structurally apart.** `simulate.py` generates only application-layer JSON
  payloads (`deviceId`/`temperature`/`soilMoisture`, matching Section III-H's own example) and
  has no import of or reference to anything network-feature-related; `provenance.py` draws only
  from held-out CICIoT2023 rows and never reads a payload field. Neither module can accidentally
  bridge the two planes, because neither has the other's data in scope.
- **`FeatureProvenanceAdapter.from_held_out_frame`, a convenience constructor that makes the
  correct construction the easy one.** Point it at the Phase 2 TEST split directly (already
  proven never-trained-on by the R3 leakage gate) and it derives provenance refs from
  `data/subsample.py`'s existing `source_file` column and the frame's own pooled-corpus index --
  no new held-out pool or bookkeeping invented. Features are returned already selected and
  scaled with the SAME fitted scaler training used, closing off train/serve skew structurally.
- **G6 verified on real data, not only synthetic fixtures.** Built the adapter from the actual
  311,573-row Phase 2 test split and paired it with a real simulated stream: sampled 200+
  distinct provenance refs and checked every one against the 1,237,958-row training index --
  **zero leaked**. This is the same "prove it on the real corpus, not just a unit test" standard
  applied earlier to the federated scaler path.
- **A real pairing bug found integrating the two modules, not caught by either module's own
  unit tests in isolation.** `TelemetryMessage` originally carried only `counter`, monotonic
  PER DEVICE (correct for its actual purpose -- Eq. 27's replay-protection AD tuple scopes
  counters per device). But the adapter's `network_features_for` took a bare int with no
  documented distinction, so pairing on `counter` gave every device's message 0 the *identical*
  held-out network record, message 1 the identical next one, and so on -- only surfaced by
  running a real simulated stream against a real adapter and inspecting the output, not by
  either module's tests alone. Fixed by adding a second field, `stream_index` (monotonic across
  the WHOLE stream, never repeating), and renaming the adapter's parameter from the scaffold's
  `message_counter` to `stream_index` so the correct call is the only obviously-named one. A
  regression test demonstrates the bug directly (pairing on `counter` collides; pairing on
  `stream_index` does not) rather than only testing the fix in isolation.
- **`true_label` added to `ProvenancedFeatures`** (flagged scaffold extension): carries the
  held-out record's ground-truth label for demo narration and test assertions only -- never
  read by anything that classifies, and the returned feature vector's shape is unchanged by it.
- **New `TelemetryConfig`** (`device_ids`, `messages_per_stream`, `schema_version`, sensor
  ranges, `seed`), following the project's typed-pydantic-config convention rather than
  scattering these as function defaults.

### Phase 4: two gaps closed after the minimal gate (FedProx wiring, scaler verification)

Audited Phase 4 after the gate closed rather than treating "gate closed" as "nothing left" --
found two real gaps neither the gate checker nor the experiment run had reason to catch, since
neither affects the minimal gate's own criterion.

- **FedProx was computed but never wired into training, until now.** `fedprox_proximal_term()`
  existed, was unit-tested, and was correctly documented as the Section III-F2 R4 fallback --
  but nothing in `federated/client.py` or `model/train.py` ever called it, so selecting it had
  no effect. Fixed: `model/train.py`'s `train_module` gained `fedprox_mu`/`fedprox_reference`
  parameters and adds the penalty to every step's loss, computed from the model's **live**
  parameters (`model.named_parameters()`, not a `state_dict()` snapshot) so the gradient
  actually reaches training rather than only being logged. `federated/client.py`'s
  `FederatedClient` gained a `fedprox_mu` constructor argument (default `None`, matching
  `configs/base.py`'s existing default -- the fallback stays opt-in, not the default regime),
  and passes the round's broadcast state as the fixed reference point.
  - **Verified behaviourally, not just "doesn't crash":** starting two identically-initialised
    models from the same point, plain training drifted 0.0207 (squared L2) from the start point
    over 8 epochs; with `fedprox_mu=10.0` it drifted only 0.0032 -- **16% of the unconstrained
    drift**, a real, substantial constraint, not a rounding-level effect. A second test confirms
    `fedprox_mu=0.0` is indistinguishable from plain training (the penalty is a true no-op at
    mu=0), and a third exercises the wiring through `FederatedClient.local_train` itself, not
    only the lower-level `train_module`. New tests in `test_training.py` (4) and
    `test_federated.py` (2) -- pytest now 251 passed / 1 skipped.
  - Not yet exercised against real data: this closes the implementation gap, not the deferred
    alpha=0.1 sweep that would be the actual test of whether R4 is needed.
- **The federated scaler-statistics path (`local_sufficient_stats`/`combine_stats`) had only
  ever been proven on synthetic data.** The real Phase 4 run reused Phase 3's already-pooled
  scaler rather than deriving it through the actual per-client-statistics mechanism, so the "no
  raw data crosses the boundary" property for scaling was demonstrated in a unit test but not on
  production data. Verified now on the real corpus: reconstructing the client partition
  actually used (alpha=0.5, same seed) over the RAW, unscaled selected features (not the
  already-standardised cache, which would have made the check vacuous -- mean already ~0), each
  client's local mean differs meaningfully from the others (e.g. `Tot sum`: 25,184 / 30,640 /
  20,991 across the three clients) and yet Chan's combination reproduces the pooled fit to
  **relative difference 2e-12** -- as exact on real, skewed production data as the synthetic
  tests already proved in principle. This was a verification exercise (manual, ~1 minute), not
  a code change; no new artifact was produced since nothing about the already-reported Phase 4
  results changes.

### Phase 4 minimal gate CLOSED: baselines 4-5, real federated experiment run

Runs `scripts/run_phase4.py` for real: 3 clients (Dirichlet α=0.5), R=20 rounds, E=3 local
epochs, weighted FedAvg, 3 seeds, against `scripts/check_phase4_gate.py`'s criterion (written
*before* this run, per the Phase 3 lesson). **25/25 checks PASS, exit 0 — `PHASE 4: MINIMAL GATE
CLOSED`.** Results written to `artifacts/manifest_phase4_default.json` (full provenance) and
`artifacts/phase4_results.json` (distilled summary, in the same relationship
`phase2_feature_selection.json` has to its own run). Total wall-clock: 440.5 minutes.

| macro-F1 | seed 0 | seed 1 | seed 2 | mean |
| --- | --- | --- | --- | --- |
| Baseline 4, client 0 (local-only) | -- | -- | -- | 0.7702 ± 0.0127 |
| Baseline 4, client 1 (local-only) | -- | -- | -- | 0.7332 ± 0.0045 |
| Baseline 4, client 2 (local-only) | -- | -- | -- | 0.7635 ± 0.0066 |
| Baseline 5 (federated global) | 0.8338 | 0.8364 | 0.8301 | **0.8334 ± 0.0026** |
| Baseline 3 (Phase 3 centralised, reference) | | | | 0.8297 |

- **G4 answered, computed rather than eyeballed** (`summarize_phase4.py`'s bracket check):
  federation beats every local-only client on every seed, **and slightly exceeds the
  centralised reference on all three** (0.8334 vs 0.8297). The paper explicitly allows the
  opposite finding (R4: federation underperforming local-only under heterogeneity is "a
  legitimate finding... the quantity an operator most needs to know"); this run did not need
  that allowance, but it was checked rather than assumed.
- **Convergence.** All three seeds' round curves climb steeply to ~round 8-10 then plateau in a
  tight 0.83-0.84 band with minor round-to-round noise (e.g. seed 0 dips to 0.811 at round 12,
  recovers next round) -- visibly converged well before R=20, a data point for judging R's
  necessity if the deferred R sweep is picked up later.
- **Partition sanity-checked against real data, not just synthetic tests.** At α=0.5 the
  per-client histograms are visibly skewed (client 1 holds 1,359 of class 1 against 12-24 of
  several others) without any client being starved to zero -- the heterogeneity knob is doing
  real work on the actual corpus, not only in `test_federated.py`'s synthetic fixtures.
- **A real logging bug found mid-run, fixed for future runs, left uncorrected in this one.**
  `federated_global_gru`'s `verbose` flag was never passed as `True` from `run_phase4.py`, so an
  entire seed's 20 rounds (order of an hour) produced no output. Fixed in `983828f`; the fix
  could not apply to the already-running process (Python had already loaded the old code), so
  seed 0's per-round progress went unobserved -- the final metrics are unaffected, only their
  visibility while running.
- **A severe, misleading timing anomaly, root-caused rather than left unexplained.** Seed 1 took
  14,807s (4.1h) against seed 0's 4,714s (1.3h) and seed 2's 4,008s (1.1h) -- a 3.1-3.7x outlier
  with no code-level explanation, since the process's own memory footprint stayed flat (~430-590
  MB RSS) throughout. Diagnosed as **system-wide memory pressure**: swap usage measured at 8.9 of
  10 GB (87%) during the slow stretch, most plausibly from other applications competing for RAM
  on the host, not from this process or its code. Recorded here so a future run's per-seed
  timing spread is not mistaken for a regression in the federated code.
- **Scope decision (flagged per Golden Rule 1, discussed with the user before this run started).**
  Section III-J4's "work plan" states Phase 4's gate as "three clients with weighted FedAvg" --
  narrower than Phase 3's, and the six ablations of Section III-I3 (α, E, weighted/unweighted,
  W, F, R) are evaluation *reporting*, not listed among the seven phase-gate criteria. This is
  the same summary-vs-source distinction that caused Phase 3 to be declared complete
  prematurely (see the correction entry below); this time `check_phase4_gate.py` was written
  first, against the paper text, specifically to not repeat it. The α/E/weighted-vs-unweighted/R
  sweep is real remaining work, deferred as a separate, much larger task (a single configuration
  alone cost 7.3 hours) -- not silently dropped.

### Phase 4 opened (user-authorised): federated infrastructure, both invariants green

Implemented `federated/scaler_stats.py`, `serialization.py`, `aggregation.py`, `partition.py`,
`client.py` and `server.py`. New `tests/test_federated.py` (23) and real bodies for the two
previously-skipped invariant tests. Pytest now **241 passed / 1 skipped** (up from 197/3) — the
only remaining skip is Phase 6's path-disjointness. **Phase 4 is not complete**: baselines 4-5
(local-only GRUs, federated global GRU) are still stubs and the federated experiment has not
been run.

- **FedAvg weight is the SEQUENCE count (Eq. 21), with a test that would fail on row counts.**
  `test_fedavg_weighting.py` constructs two clients holding equal rows but unequal run
  structure, so sequence-weighting gives 1.0 and the row-count bug gives 5.0 — the invariant is
  checked by the two answers differing, not merely by the right one appearing.
- **Chan's combination is exact against the pooled fit (Eqs. 23-24).** `test_scaler_equivalence`
  compares deliberately *unequal* client shares (an equal split would pass even with an
  unweighted mean-of-means, hiding the n_k weighting), repeats over ten random splits, and
  carries a negative control proving the weighting does real work. A further test uses features
  with mean ~1e8 and spread ~1, where the `E[x^2] - E[x]^2` route cancels catastrophically —
  which is why the transmitted triple is `(count, mean, M2)` rather than the prose's
  "count, sum, and sum of squares".
- **Eq. (22)'s communication figures reproduce exactly**, and are checked against the bytes this
  implementation actually serialises rather than trusted: |θ| = 33,800 → **811,200 B = 0.7736
  MiB per round** (paper: ~0.77 MiB), **15.47 MiB over 20 rounds** (paper: ~15.5 MiB), against
  276 MB to pool the records — a factor of **17.0** (paper: "roughly 18").
- **Per-class Dirichlet draws, not one draw reused (flagged per Golden Rule 1).** Eq. (20) is
  written per class; drawing once would give every client the same share of every class — a
  uniform partition wearing a Dirichlet's clothes, and exactly the L2 under-specification
  Section III-F1 exists to avoid. A test asserts client 0's share differs across classes, and
  another asserts large α is measurably more uniform than small α, so the sweep knob is known
  to do something.
- **Client independence is enforced, not asserted (A1).** Local data is private with no public
  attribute (a test checks the public surface), the broadcast global state is cloned before use
  so a client cannot mutate the server's tensors, and a **fresh optimiser is built every round**
  — carrying or aggregating Adam moments would make the method not FedAvg while still appearing
  to converge.
- **Every update crosses the boundary through safetensors even in-process**, so the no-pickle
  rule is structurally true rather than a claim about code that would transmit differently if
  wired to a socket. `n_k` travels in the safetensors *header metadata* rather than as a tensor
  smuggled into the state dict, so a received state loads into a module without first stripping
  a bookkeeping key; a missing count is an error, never a silent zero, since it is the
  aggregation weight.
- **A client holding zero sequences is legitimate, not an error.** Under α = 0.1 a client can
  receive too few blocks to form a single window at W = 16; it contributes weight 0 and appears
  in the published histogram. Raising there would make a declared experimental configuration
  unrunnable.
- **Scaffold inconsistency reconciled:** `aggregation.StateDict` was a `Mapping` while
  `serialization.StateDict` was a `dict`, so the two modules could not compose without a cast.
  Aggregation now accepts the wider `Mapping` and returns a concrete `dict`.

### Changed: run manifests are now version-controlled

`artifacts/` was gitignored wholesale, so every manifest and phase report existed only on the
machine that produced it. That contradicts Section III-I4 ("no headline number is reported
without a manifest"): the figures quoted throughout this file were claims with their evidence
sitting outside the repository, and losing one laptop would have meant re-deriving them (114
minutes for the Phase 3 run alone). Manifests and phase reports are provenance, not bulk output
— all five total ~200 KB, and each records the git commit it belongs to, so the repo is exactly
where they belong. Model checkpoints, figures and scratch output stay ignored.

Note for anyone editing the rule: the pattern is `artifacts/*`, **not** `artifacts/`. Git never
descends into an excluded *directory*, so a `!` re-include beneath one is silently ignored —
the first attempt at this change looked correct and tracked nothing.

### Phase 3 gate CLOSED — baselines 1-3 and the window ablation

Answers the correction below. `eval/baselines.py` implemented (baselines 1-3; 4-5 stay Phase 4
stubs, with a test asserting they still raise), plus `scripts/run_phase3_complete.py` and
`scripts/check_phase3_gate.py`. New `tests/test_baselines.py` (10). Run: 3 baselines × W ∈ {1,16}
× 3 seeds × 10 epochs, 113.8 min, manifest at
`artifacts/manifest_phase3_complete_default.json` (gitignored).

| macro-F1 | W=1 | W=16 |
| --- | --- | --- |
| random forest | 0.6884 ± 0.0016 | 0.6855 ± 0.0006 |
| MLP (parameter-matched) | 0.6097 ± 0.0033 | 0.6070 ± 0.0020 |
| **centralised GRU** | 0.5974 ± 0.0045 | **0.8297 ± 0.0013** |

At W=16 the GRU also reports balanced accuracy 0.8805 ± 0.0047, MCC 0.8785 ± 0.0024 and
accuracy 0.9030 ± 0.0022.

- **Ablation answered (Section III-D).** The paper set the test: "if it matches W = 16,
  recurrence has not earned its place in the architecture, and we will say so." It does not
  match — W=16 beats W=1 by **+0.2322 macro-F1, about 51× the seed standard deviation**. The
  verdict is computed by the runner (gain vs. 2× seed std) and recorded in the manifest, so it
  could not be rationalised after the fact. **Recurrence earned its place.**
- **S13's dissent was a real threat and was answered.** The random forest genuinely beats both
  neural models at W=1 (0.6884 vs GRU 0.5974) — so the baseline was doing work, not
  rubber-stamping the architecture — but the GRU clears it by **+0.144** once it has a full
  window. Notably the RF is nearly window-invariant (0.6884 → 0.6855), as it must be, since it
  only ever sees one record.
- **The MLP comparison isolates recurrence cleanly.** Parameter-matched to 33,808 against the
  GRU's 33,800 (0.02% apart) and trained through the *same* `train_module` loop — same
  optimiser, class-weighted CE, seeding, batching, epochs — so the **+0.2227** gap at W=16 is
  attributable to recurrence and not to capacity or training setup. Its near-identical scores at
  W=1 and W=16 (0.6097 / 0.6070) confirm it really is blind to history, as designed.
- **Recurrence helps most exactly where it is needed most.** Per-class F1 at W=16, GRU vs RF:
  BruteForce **0.662 vs 0.264**, WebBased **0.601 vs 0.345**, Benign **0.736 vs 0.486**. The
  rare families and the benign class — the ones that carry the operational cost — are where the
  window pays off; on the easy flooding classes the two are close (Mirai 0.9985 for both).
- **Longer training mattered, as flagged.** 10 epochs lifted the W=16 GRU from the earlier
  3-epoch 0.7866 to 0.8297 (+0.043), confirming that run was under-trained rather than at the
  architecture's ceiling.
- **Timing mis-estimated twice, recorded so the method is not repeated.** The first estimate
  extrapolated a random-forest benchmark run on *random labels*, which is pathological for trees
  (they grow to full depth memorising noise) and overstated RF cost ~10×. The second
  "correction" was worse: the MLP's seed-0 fit took 1455s against 62s for every other seed,
  because the test suite, ruff, mypy and the gate checker were run **on the same machine during
  that fit**. A timing taken under self-inflicted load was reported as the model's cost. Results
  were unaffected (contention changes wall-clock, not arithmetic); the estimate was not.
- **`scripts/check_phase3_gate.py` makes the gate machine-checkable** rather than a judgement
  call: it reads the manifest and checks seeds, mean ± std for every headline metric, all three
  baselines, per-class F1 and confusion matrices, the ablation verdict, and manifest provenance,
  exiting non-zero on any failure. It deliberately does **not** check that the detector is good —
  a poor result properly measured closes the gate; a good result improperly measured does not.
  Validating it against the *older* manifest (which it should reject) exposed two bugs in the
  checker itself — a `TypeError` crash on a flat-summary manifest, and mistaking the
  `per_class_f1` map for a window key — both fixed before it was trusted.

### Still open after Phase 3

- **Binary FPR remains too high to deploy** (0.257 ± 0.061 at the 3-epoch run; not re-measured
  here, since the completion run reports the multiclass bundle). Tuning the operating point is
  outstanding work, not a Phase 3 blocker.
- **The window sweep is partial.** Section III-I3 specifies W ∈ {1, 8, 16, 32}; only the
  decision-relevant endpoints W=1 and W=16 were run. W=8 and W=32 remain, as does the F sweep.
- The detection-latency set (below) is still unresolved.

### Correction (same day): Phase 3's gate was declared clear too early

An earlier version of this entry and of the phase table above recorded Phase 3's exit criterion
as **met** on the strength of the metric protocol alone. That was wrong, and is corrected here
rather than quietly edited away. **Section III-I1 requires five baselines "reported for every
configuration"**, three of which the scaffold itself tags Phase 3 in
`eval/baselines.py`'s `NotImplementedError` messages:

1. random forest on single records — **still a stub** (and the most pointed omission: S13 found
   a random forest strongest on tabular flow features, which is precisely why the paper carries
   this baseline — it is the one that can contradict the GRU choice);
2. MLP on single records — **still a stub** (it is what isolates the contribution of recurrence);
3. centralised GRU — done, though produced via `scripts/run_phase3.py` rather than through
   `eval/baselines.centralized_gru`'s entry point, which remains a stub.

Baselines 4 and 5 (local-only GRUs, federated global GRU) are correctly Phase 4.

Additionally, Section III-D states the W sweep includes **W = 1 as an ablation**: "if it matches
W = 16, recurrence has not earned its place in the architecture, and we will say so." That
comparison has not been run, so the architecture's central claim is so far unexamined on this
data. Phase 3 therefore remains **open**, and Phase 4 stays gated behind it.

### Open question raised, not resolved (blocks part of Section III-D)

- **The detection-latency evaluation set cannot be built as specified.** Section III-D requires
  "a separate held-out set of mixed windows spanning a benign-to-attack boundary … used to
  measure detection latency", *and* that windows be constructed "only over contiguous same-label
  runs within a single source file". On the raw UNB distribution these cannot both hold:
  **every capture file contains exactly one label** (verified — max labels per `source_file` is
  1), because the benchmark names each capture after its attack type. A benign→attack transition
  inside one source file therefore does not exist anywhere in the corpus, and
  `build_latency_eval_windows` correctly returns **zero** windows on the real data. The three
  constraints (mixed benign→attack windows, single-source-file windows, one-label-per-file data)
  are mutually unsatisfiable. Resolving it requires a user decision — synthesise transitions by
  concatenating a benign run to an attack run from a different file (fabricates a transition
  that never occurred, the same objection that rules out synthetic oversampling), relax the
  single-source-file rule for the latency set only, or drop the latency metric and report why.
  Nothing is implemented for it pending that decision.

- **Phase 2 (four-stage feature selection) implemented — Phase 2 is now complete.**
  `features/selection.py`'s `FeatureSelector.fit`/`.transform` implement all four stages of
  Section III-C, fitted on training blocks only. New `tests/test_feature_selection.py`
  (18 tests). Pytest now 102 passed / 4 skipped (up from 84 passed / 4 skipped).
  - **Stage 4 needs Phase 3, which is gated behind Phase 2 — a circular dependency in the
    paper's own phase ordering, flagged rather than resolved by breaking the gate.** Choosing
    "the knee of the validation macro-F1 curve" (Stage 4) requires training and evaluating a
    detector at each candidate F, but the detector is Phase 3 and Phase 3 is the hard gate that
    opens only once Phase 2 closes. Resolved by dependency injection instead of starting Phase 3
    early: `fit()` takes an optional `evaluator` callable mapping a candidate column list to a
    validation macro-F1. Supplied (Phase 3 onward), Stage 4 runs for real and records the curve
    in `SelectionResult.f_sweep_scores`; `None` (Phase 2, now), it falls back to the configured
    `f` and records an **empty** curve rather than fabricating one — so no GRU is trained before
    its gate. The knee itself is a standard maximum-distance-to-chord construction and is unit
    tested against a deliberately placed knee.
  - **The reciprocal-rank-fusion equation is not recoverable from the paper source (flagged,
    Golden Rule 1).** `docs/design_paper.md`'s PDF extraction dropped *every* display equation
    before Eq. (12) — which includes Stage 3's RRF formula and the mutual-information definition.
    The surrounding prose survives, the equations do not. Implemented as the canonical
    `RRF(j) = sum_m 1/(k + r_m(j))` with 1-based ranks and `k = 60` (Cormack et al., 2009),
    exposed as the new `FeatureConfig.rrf_k`. **This needs re-checking against the authoritative
    PDF before any Stage-3 number is quoted in a write-up** — the missing equations are a
    repo-wide source-fidelity gap, not just a Stage-3 one.
  - **Two further undefined terms, resolved and documented:** "univariate relevance" (Stage 2's
    tie-break for which member of a correlated pair survives) is read as mutual information with
    the label — the same statistic Stage 3 ranks by, so the two stages share one notion of
    relevance; and "disagreement" (Stage 3) is reported as the selection-relevant one, a feature
    in the top-F of exactly one of the two rankings, which is precisely what the fusion step
    would otherwise bury.
  - **Statistics computed on a seeded 200,000-row sample** of the training split (new
    `FeatureConfig.selection_sample_size`; `None` uses every row), shared by Spearman, mutual
    information and the random forest so all three see identical rows. Measured on this machine:
    MI and RF each cost ~16s per 100k rows and scale linearly, so a full-split fit would run to
    minutes for estimates already stable at this size; the sampled fit takes ~63s.
  - **Non-finite rows excluded from the statistics sample only, never imputed or zero-filled**,
    and the count reported in `SelectionResult.n_nonfinite_rows_excluded` — same policy
    `data/characterize.py` already applies to its correlation accumulation, and the reason is
    the genuine `+inf` in `Rate` that Phase 1 found in the raw corpus.
  - **Scaffold extension (flagged):** `SelectionResult` gained `fused_ranking`,
    `f_sweep_scores`, `selected_f` and `n_nonfinite_rows_excluded`. Section III-C requires the
    fused ranking and the Stage-4 knee curve to be reportable, and the stub dataclass had no
    field for either; the non-finite count is added on the same principle as Phase 1's
    `infinite_counts` (visible, not silent).
  - **Real results on the real training split** (1,237,958 rows; saved to
    `artifacts/phase2_feature_selection.json`, gitignored): Stage 1 dropped `source_file` and no
    zero-variance column (consistent with Phase 1's finding that the raw corpus, unlike the
    Kaggle mirror, has none). Stage 2 pruned **8** columns at tau=0.95 — `Rate`, `syn_count`,
    `fin_count`, `rst_count`, `ARP`, `IPv`, `AVG`, `Std` — leaving **F0 = 31**. 54 non-finite
    rows excluded. The selected F=16 set, in fused-rank order: `Tot sum`, `Header_Length`,
    `Tot size`, `Max`, `Variance`, `IAT`, `TCP`, `ack_count`, `Time_To_Live`, `syn_flag_number`,
    `ack_flag_number`, `UDP`, `Protocol Type`, `psh_flag_number`, `fin_flag_number`, `Min`.
    **The two rankings genuinely disagree on 4 features** (`fin_flag_number`, `ICMP`, `Min`,
    `Number`) — reported, per Section III-C, rather than averaged away; note mutual information
    ranks `IAT` outside its top 8 while the random forest ranks it *first*, the clearest instance
    of the non-monotonic-vs-high-cardinality split the paper predicts.

- **Phase 2 (leakage control), subsampling + label derivation implemented; the full
  SUBSAMPLE -> DEDUP -> SPLIT pipeline now runs end-to-end on real data.**
  `data/subsample.py`'s `stratified_capped_subsample` is implemented (Section III-B1, risk R1).
  New `tests/test_subsample.py` (10 tests, synthetic fixtures). Pytest now 84 passed / 4 skipped
  (up from 74 passed / 4 skipped).
  - **Dataset root decision (flagged per Golden Rule 1, confirmed with the user via an explicit
    choice, not silently resolved).** `configs/base.py`'s `DataConfig.dataset_root` and
    `configs/default.yaml` now point at the official UNB raw distribution
    (`data/ciciot2023_raw/`), not the pre-merged Kaggle mirror (`data/ciciot2023/`) used
    previously. Three things in the paper text make this the only choice consistent with
    Sections III-B/D, laid out for the user before the choice was made: (1) Section III-B1 says
    subsampling reads "whole CSV parts in fixed order" -- a description of many per-capture
    files, which only the raw distribution has; (2) Section III-D requires windows to stay
    "within a single source file", recoverable only when each CSV *is* one capture, as in the
    raw distribution -- the Kaggle mirror had already destroyed that identity by merging; (3) the
    raw distribution's record count (46,776,700) matches the paper's stated ~46.7M almost
    exactly, confirming it -- not the Kaggle mirror's ~17% subsample -- is the benchmark the
    paper describes. The Kaggle mirror is no longer consumed anywhere in the pipeline; it remains
    on disk for reference only.
  - **Label derivation (flagged per Golden Rule 1).** The raw distribution has no `label` column;
    the label lives only in each CSV part's *filename*. Directory name and filename disagree for
    benign traffic (`Benign_Final/` contains `BenignTraffic*.pcap.csv`), so labels are derived
    from the filename (strip trailing `.pcap.csv`, then a trailing digit run, then one leftover
    `-`/`_`), not the directory. This reproduces all 34 of the raw distribution's implied
    classes and matches the Kaggle mirror's own 34-label vocabulary almost exactly (e.g.
    `DDoS-PSHACK_Flood`, `DDoS-RSTFINFlood`), a useful cross-check even though the Kaggle mirror
    itself is no longer used as data.
  - **Per-class cap mechanic (flagged per Golden Rule 1).** Parts are read in natural
    part-number order (`Foo.pcap.csv`, `Foo1.pcap.csv`, `Foo2.pcap.csv`, ... -- lexicographic
    order would wrongly visit `Foo10` before `Foo2`) and a part, once started, is always read to
    completion (`chunk_size` bounds memory *within* one part's read; it never stops a read
    mid-part). Reading stops once a class's accumulated rows reach or exceed `per_class_cap`.
    Because some CICIoT2023 attack-type directories hold parts far larger than a typical cap (a
    single `DDoS-ICMP_Flood` part alone is ~268k rows against the old 200k cap), whole-part
    reading alone would let the cap overshoot by up to one whole part per class. To keep "cap" a
    real ceiling, a *seeded, order-preserving* random trim (no shuffling -- kept rows retain
    their original relative order) drops any excess down to exactly `per_class_cap` once a
    class's accumulated rows exceed it; this is the only place `seed` is used, since read order
    and per-part inclusion are otherwise fully deterministic. `target` is not used to derive
    `per_class_cap` -- the paper does not specify that relationship, and the two are independent
    config knobs -- it is only a sanity check (`warnings.warn` if the realised total deviates
    from it by more than 50%).
  - **Source-file identity added, resolving the gap flagged in the 2026-09-13 `dedup.py` entry.**
    Every row now carries a `source_file` column (the part's path relative to `dataset_root`), so
    Section III-D's "never window across a source file" rule is enforceable once
    `sequences/windowing.py` is implemented (Phase 2/3 boundary).
  - **Config recalibration, a real finding surfaced rather than silently patched over.** Running
    the implemented subsampler against the real raw distribution with the inherited
    `per_class_cap: 200_000` (tuned, it turns out, for the Kaggle mirror's per-class
    distribution, never re-validated against the raw one) produced 4,458,051 records -- more
    than double the paper's stated M ~ 1.5-2e6 (Section III-B1), because 19 of 34 classes hit
    that cap on the raw distribution, not just the "dominant DDoS classes" the paper names (also
    `BenignTraffic`, `MITM-ArpSpoofing`, `VulnerabilityScan`, ...). Swept `per_class_cap` against
    the real data and reset the default to **70,000**, which caps 24 of 34 classes and lands the
    realised total at **1,772,371** records, inside the paper's target range.
    `configs/base.py`/`configs/default.yaml` updated with the new default and this rationale
    recorded inline.
  - **Full pipeline validated end-to-end on real data (not just synthetic fixtures).** Chaining
    the new `subsample.py` with the existing `dedup.py`/`split.py` against
    `data/ciciot2023_raw/` under the recalibrated config: 1,772,371 records subsampled ->
    1,549,531 after dedup (222,840 exact duplicates removed, a 12.6% duplicate rate within this
    subsample) -> 6,072 label-respecting blocks -> stratified split (4,854 train / 1,218 test
    blocks, 1,237,958 / 311,573 rows) -> the R3 leakage assertion (no record hash in both
    partitions) **passes on real data**, not only on the synthetic fixture `test_leakage.py`
    already covered. This was a manual verification run, not a new automated test (the real
    dataset is gitignored and not available in CI-equivalent local runs), but it is the first
    time the R3 gate's *intent* -- not just its synthetic proxy -- has been confirmed.

## 2026-09-13

### Added

- **Phase 2 (leakage control), dedup + block-split half implemented; the R3 gate is now
  active.** `data/dedup.py`'s `record_hash`/`deduplicate` and `data/split.py`'s
  `make_blocks`/`stratified_block_split` are implemented, and `tests/test_leakage.py` has been
  rewritten from its skipped placeholder into a real gating test: it deduplicates synthetic data
  containing planted triple-duplicate rows, block-splits the result, and asserts the train/test
  hash sets are disjoint -- plus a negative-control test
  (`test_leakage_would_occur_without_dedup`) proving the same data *does* leak when split
  without deduplicating first, so the gate is not vacuously true. New `tests/test_dedup.py`
  (5 tests) and `tests/test_split.py` (8 tests). Pytest now 74 passed / 4 skipped (down from
  5 skipped: `test_leakage.py` no longer skips).
  - **`record_hash`**: a single 64-bit `pandas.util.hash_pandas_object` hash per row rendered as
    hex, matching the collision-probability reasoning already documented for Phase 1's
    `characterize.py`. Deliberately scoped to the capped-subsample regime (~1.5-2e6 rows, see
    below), where the birthday-bound collision risk is ~1e-10, not the full 46.7M-row corpus.
  - **`deduplicate`**: keeps first occurrence by that hash, preserves row order (never shuffles),
    resets the index to a clean contiguous range.
  - **`make_blocks`**: assigns contiguous block ids, breaking whenever `block_size` rows have
    accumulated OR the `label` column changes -- a resolution of something the paper leaves
    implicit (see below).
  - **`stratified_block_split`**: draws `test_fraction` of each label's blocks into a global test
    set via a seeded RNG, raises if any block is found to span more than one label (defensive;
    `make_blocks` itself never produces one).
  - **Ordering decision, flagged rather than left implicit (Golden Rule 1):** the paper's R3
    invariant fixes dedup *before split*, but says nothing about dedup's position relative to
    subsampling. `deduplicate`'s signature (`frame: pd.DataFrame` in memory) only fits the
    ~1.5-2e6-row capped subsample of Section III-B1, not the 46.7M-row raw corpus (which does not
    fit as one in-memory frame — Phase 1 handles that scale by streaming instead). This fixes the
    pipeline order as **SUBSAMPLE -> DEDUP -> SPLIT**, documented in `dedup.py`'s module
    docstring. Not yet run on real data for this reason: `data/subsample.py` (needed to produce
    a memory-feasible, real, labeled frame from `data/ciciot2023_raw/`) is still a stub, as is
    label derivation from the raw distribution's per-attack-type directory names.
  - **Block-boundary decision, flagged (Golden Rule 1):** `make_blocks` never lets a block span a
    label change, so blocks can be stratified by a single label in
    `stratified_block_split`. The paper doesn't specify block-construction mechanics explicitly;
    this anticipates Section III-D's later, explicit rule that *windows* may never cross a label
    boundary either, so blocks (the coarser unit windows are drawn from) are made to agree with
    that constraint rather than risk being finer-grained now and re-litigated at windowing time.
  - **Gap surfaced, not yet resolved:** neither this module nor `characterize.py` tracks which
    original source *file* each pooled row came from. Section III-D requires windows to stay
    within "a single source file" even when labels happen to match across files, so whatever
    ingests the raw distribution into one ordered frame (likely `data/subsample.py`, not yet
    implemented) will need to carry a source-file identity column through dedup, split, and
    windowing. Flagged here so it isn't lost before that module is written.
- **Phase 1 (dataset characterisation), implemented and run on real data.**
  `data/characterize.py`'s `characterize_dataset` now streams every `*.csv` file found under
  `dataset_root` in fixed-size chunks (`pandas.read_csv(chunksize=...)`) and reports: exact
  columns/dtypes, per-column null counts, zero-variance columns (min == max among observed
  values), an exact duplicate-row count (order-sensitive 64-bit row hash via
  `pandas.util.hash_pandas_object`; birthday-bound collision probability ~1e-6 at this corpus
  size, documented in the module docstring), label vocabulary + counts, the class imbalance
  ratio, and a Pearson correlation matrix over numeric columns (streamed via
  running sum/sum-of-squares/sum-of-products, one pass, O(k^2) memory). New
  `tests/test_characterize.py` (5 tests, synthetic fixtures via `tmp_path`) — pytest now
  57 passed / 5 skipped.
  - **Scaffold gap found and fixed (flagged per golden rule 1, confirmed with the user):** the
    stub `CharacterizationReport` dataclass had no field for the correlation matrix, even though
    Section III-B requires one in the Phase 1 report. Added `correlation_matrix: dict[str,
    dict[str, float]]`. Pearson was chosen over Spearman so the full pooled corpus can be
    covered exactly in one streaming pass; this is a separate statistic from Phase 2's Spearman
    correlation *pruning* (Section III-C, Stage 2, tau=0.95), which is training-data-only and
    feeds feature selection, not this report.
  - **Dataset provenance conflict found and resolved (flagged per golden rule 1, confirmed with
    the user):** the CICIoT2023 data supplied for this project (`data/ciciot2023/{train,test,
    validation}/*.csv`) arrived already split by an external mirror builder, and is a ~7.85M-row
    subsample (34 labels, 47 columns, schema matches the paper) rather than the paper's stated
    ~46.7M-record raw corpus. Consuming that external split as-is would make Section III-B2/R3
    (dedup must precede *our* split) unverifiable, since we have no visibility into how or
    whether the mirror deduplicated before assigning rows to train/test/validation. Resolution:
    `characterize_dataset` pools every CSV under `dataset_root` (recursively) into one corpus
    regardless of which subdirectory it came from, and Phase 2 will dedup + split that pooled
    corpus ourselves, ignoring the external split assignment. This also generalises correctly to
    the real CICIoT2023 distribution format (many per-capture CSV parts, no pre-existing split).
  - **Real numbers produced on the pooled corpus** (saved to
    `artifacts/phase1_characterization_report_kaggle.json`, gitignored): 7,845,673 records, 47
    columns, 0 nulls anywhere, zero-variance columns `Telnet` and `IRC`, 275,259 exact duplicate
    rows (confirms the paper's claimed leakage channel is real in this data), 34 labels,
    imbalance ratio ≈ 5,820 (paper states IR ≈ 5,751 on the full corpus; the difference is
    expected since this is a subsample, not the raw benchmark).
- Development environment stood up on this machine (`.venv`, `pip install -e ".[dev]"`,
  `pre-commit install`) — CLAUDE.md had assumed one already existed. All four gates (`ruff
  check`, `ruff format --check`, `mypy src`, `vulture`) confirmed clean before and after this
  work.
- **Second dataset added: the official UNB raw CICIoT2023 distribution** (`data/ciciot2023_raw/`,
  309 per-capture CSV files, 8.3 GB, one directory per attack type — the format the paper
  actually describes, as opposed to the pre-merged Kaggle mirror above), and `characterize_dataset`
  run on it too, at the user's request, specifically to diff it against the Kaggle mirror.
  - **`.gitignore` gap fixed:** added a blanket `/data/` rule. The prior `*.csv`/`*.parquet`-only
    rules would have let non-CSV dataset files (e.g. this distribution's `README_CSV.pdf`) get
    committed, contradicting the section's own "must not be committed" comment.
  - **Correctness bug found and fixed while running Phase 1 on this data.** The correlation
    accumulator filtered `NaN` but not `+/-inf` out of the numeric block before accumulating
    sums; the raw distribution's `Rate` column (a division by flow duration) has genuine `+inf`
    values for near-zero-duration flows, which silently corrupted the running covariance
    (surfaced as a `RuntimeWarning: invalid value encountered in subtract`, not a hard failure —
    it would have shipped a wrong correlation matrix unnoticed). Fixed the mask to
    `np.isfinite(...).all(axis=1)`, and added a new `infinite_counts: dict[str, int]` field
    to `CharacterizationReport` (parallel to `null_counts`) so infinite values are reported
    rather than silently dropped. New regression test
    `test_characterize_infinite_values_counted_and_excluded_from_correlation` — pytest now
    58 passed / 5 skipped.
  - **Real numbers on the raw corpus** (saved to
    `artifacts/phase1_characterization_report_raw.json`, gitignored): **46,776,700 records** —
    matches the paper's stated ~46.7M almost exactly, confirming the Kaggle mirror is indeed a
    ~17% subsample rather than the full benchmark. 39 columns, **no `label` column at all**
    (labels are implicit in the per-attack-type directory/file names in this distribution, unlike
    the Kaggle mirror which already merged in a `label` column). 0 zero-variance columns (vs. the
    Kaggle mirror's `Telnet`/`IRC` — the fuller corpus has at least one nonzero value for both).
    **25,641,443 exact duplicate rows — 54.8% of the corpus**, dramatically higher than the
    Kaggle mirror's 3.5%; this is the leakage channel Section III-B2/R3 exists to close, and its
    scale on the real corpus underscores why dedup-before-split is a hard invariant here, not a
    nicety. Small genuine null counts in several columns (the Kaggle mirror had zero), and 1,037
    `+inf` values in `Rate`.
  - **Schema diff (columns only in one dataset):** Kaggle-only —
    `flow_duration, Duration, Srate, Drate, urg_count, Magnitue, Radius, Covariance, Weight,
    label` (10 columns, all absent from the raw distribution); raw-only — `Time_To_Live, IGMP`
    (2 columns, absent from the Kaggle mirror). 37 columns are common to both. This means the
    Kaggle mirror is not merely a subsample: it was built with extra derived columns and an
    injected `label` column that the official raw distribution does not carry, so Phase 2's
    feature selection and label derivation will need to target whichever dataset is used as the
    project's source of truth — **not yet decided, flagged to the user**.

## Unreleased

### Added

- **Phase 6 (Ascon-AEAD128 crypto core), gate deliberately overridden.** Implemented
  `crypto/ascon_aead.py` (`AsconAEAD128` encrypt/decrypt facade, `NonceRegistry`,
  `NonceReuseError`, `verify_kat_conformance`, `backend_provenance`) over a vendored official
  reference backend. **This started Phase 6 substantive logic while Phase 3 (the hard gate) and
  Phases 1, 2, 4, 5 are still unmet** — CLAUDE.md golden rule 2 was consciously waived by the
  user for this work, and is recorded here so the out-of-order start is auditable. The alerting
  path / path-disjointness half of Phase 6 (`routing/alert_sink.py`, `routing/cloud_sink.py`,
  `test_path_disjointness.py`) is **not** done, so the phase is *in progress*, not complete.
  - **Backend selection (KAT-gated, III-G1/R5).** The PyPI `ascon` candidate was rejected: only
    version `0.0.9` is published (the previous `>=1.3` pin was unsatisfiable) and it implements
    only the pre-standard Ascon v1.2 variants (`Ascon-128`/`Ascon-128a`), never SP 800-232
    `Ascon-AEAD128`. Per the paper-consistent fallback, the official reference implementation
    (`meichlseder/pyascon`, commit `ed24e54`, CC0) is vendored verbatim under
    `src/ascon_smart_agri/crypto/_vendor/pyascon/` (with `LICENSE` and `PROVENANCE.md`).
  - **KAT gate passes:** all 1089 official SP 800-232 vectors (from `ascon/ascon-c`, committed
    at `tests/kat/LWC_AEAD_KAT_128_128.txt` with `PROVENANCE.md`) verify for both encryption and
    decryption. `test_ascon_kat.py`, `test_ascon_tamper.py`, and `test_nonce_collision.py` are
    activated (skips removed) and green — pytest now 41 passed / 5 skipped.
  - **Associated-data byte encoding — decided AND implemented (golden rule 1 waived by the
    user).** Eq. (27) defines the AD tuple but no byte serialisation. The user waived golden
    rule 1 for this specific decision and chose a **length-prefixed (TLV-style)** encoding over
    fixed-width (fixed-width can silently re-introduce the cross-device collision via
    truncation/padding, and needs id-length figures Phase 1 hasn't produced). The decision, exact
    byte layout, API, and test plan are recorded in `docs/plans/phase6-ad-serialization.md`.
    `AssociatedData.to_bytes()`/`.from_bytes()` now implement it — `fmt_version:u8=0x01 ||
    L(edge_id):u16 || edge_id || L(device_id):u16 || device_id || counter:u64 ||
    L(schema_version):u16 || schema_version`, big-endian, UTF-8, injective and canonical.
    `to_bytes` raises on a >65535-byte field or out-of-u64 `counter` (never truncates);
    `from_bytes` is a strict single-valid-encoding parse (rejects unknown `fmt_version`,
    truncated input, and trailing bytes). The AEAD facade stays decoupled — it still operates on
    `bytes` and callers pass `associated_data.to_bytes()`. `backend_provenance()` now records
    `"ad_encoding": "length-prefixed-v1"` for the run manifest, and the module docstring's
    "UNRESOLVED SPEC GAP … DEFERRED" block is replaced with the decided encoding, noted as an
    intentional extension of the paper. New `tests/test_ascon_ad_encoding.py` (round-trip incl.
    empty/non-ASCII/u64-max, the anti-ambiguity and truncation-aliasing cases, neighbouring-AD
    decryption → `⊥`, strict-parse rejections, bounds, and exact wire layout) — pytest now
    52 passed / 5 skipped.
  - **Tooling:** removed the unsatisfiable `crypto = ["ascon>=1.3"]` extra and the `ascon.*`
    mypy override; excluded the vendored code from ruff, mypy (strict), and vulture so it stays
    byte-identical and diff-verifiable against upstream.
- `docs/plans/phase6-replay-window.md`: a planning doc for the receiver-side replay-window
  enforcement that consumes the authenticated `counter` (the counterpart to
  `phase6-ad-serialization.md`, which scoped replay *out* of the crypto core). Records where the
  check belongs (`routing/cloud_sink.py`, after AEAD verification, never in the crypto core),
  the path-disjointness constraint, a proposed `ReplayGuard` shape, and a test plan.
  Planning/research only — the core policy choices (per-device vs. per-key scope, strict-monotonic
  vs. sliding window, persistence) are **unspecified by the paper and must be surfaced to the
  user per golden rule 1 before implementation**; nothing is implemented. The `AssociatedData`
  docstring in `crypto/ascon_aead.py` now states replay enforcement is out of scope for that
  module and points here.
- `CHANGELOG.md` (this file), tracking repository status per phase.
- `CLAUDE.md` golden rule 6, requiring `CHANGELOG.md` to be kept current after every change.
- `docs/plans/phase6-ascon-implementation.md`: a Phase 6 planning doc for the Ascon-AEAD128
  backend decision — library research (PyPI `ascon` package vs. vendoring the official
  `meichlseder/pyascon` reference implementation), KAT vector sourcing (`ascon/ascon-c`,
  NIST's ACVP-Server), and a proposed step-by-step implementation sequence. Linked from
  `CLAUDE.md`'s "Open decision" section. Planning/research only — no crypto logic implemented;
  Phase 3's hard gate is still unmet and Phase 6 has not started.

### Repository state as of 2026-09-12

- **Mostly scaffolding.** All source files under `src/ascon_smart_agri/` are typed stubs
  (function/class signatures + docstrings citing the relevant design-paper section or equation)
  **except** the Ascon-AEAD128 crypto core `crypto/ascon_aead.py` and the vendored backend under
  `crypto/_vendor/pyascon/`, which are now implemented. Otherwise no working logic beyond
  `__init__.py` files and the `Array` type alias in `_types.py`.
- **Tests:** `pytest` reports 52 passed, 5 skipped, 0 failed. The 5 remaining skips are explicit
  `pytest.skip("pending Phase N: ...")` markers on tests that require unstubbed logic
  (`test_fedavg_weighting.py`, `test_leakage.py`, `test_model_param_count.py`,
  `test_path_disjointness.py`, `test_scaler_equivalence.py`). The Ascon KAT/tamper/nonce and
  AD-encoding tests are active and green.
- **Quality gates:** `ruff check`, `ruff format --check`, `mypy src`, and `vulture` all pass
  clean on the current tree.
- **Config scaffold:** typed pydantic config (`configs/base.py`, `configs/default.yaml`) exists
  and validates; sweep values for the six ablations are not yet exercised by any run.
- **Open decision resolved:** the Ascon backend fallback is settled — the PyPI `ascon` candidate
  is non-conformant (v1.2-only) and unsatisfiable at the pinned version, so the official
  `pyascon` reference implementation is vendored and gated by the KAT (see the Phase 6 entry
  above).

## 2026-09-12 — `1c446ee`

- Added `data/` to `.gitignore` (dataset artifacts are not checked into version control).

## 2026-09-11 — `8944683`

- Initial commit: project scaffolding — typed stub modules for all seven phases, pydantic
  config, pre-commit gate configuration (ruff, mypy, vulture, pytest), test suite with
  phase-gated skips, `README.md`, `AGENTS.md`, `CLAUDE.md`, and `docs/design_paper.md`.
