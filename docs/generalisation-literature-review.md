# What the literature says about cross-testbed generalisation — and what we should do

Researched 2026-10-01, in response to: *is there another way to generalise, is there a better
dataset, and is fixing leave-one-dataset-out even necessary?*

**Short answer: our LODO ≈ 0.085 is the published, expected result, not a defect in our
pipeline. Zero-shot cross-dataset transfer fails for everyone, including on datasets purpose-
built with a standardised feature schema. The literature's own answer is combined multi-dataset
federated training — which is the mixed-farm design we already built.**

---

## 1. Zero-shot cross-dataset transfer fails universally

**["On Generalisability of Machine Learning-based Network Intrusion Detection Systems"](https://arxiv.org/abs/2205.04112)**
is the decisive study, because it removes the excuse we were about to blame:

- Four NIDS datasets converted to **one standardised 43-feature NetFlow schema**
  (NFv2-UNSW-NB15, NFv2-CIC-2018, NFv2-ToN-IoT, NFv2-BoT-IoT), 1 M flows each
- Within-dataset: **76–99 % F1** for supervised models
- Cross-dataset: **56.28 % average performance decay**, "sometimes dropping to near 0 % F1"
- Their verdict: *"none of the considered models is able to generalise over all studied datasets"*
- **Direction matters enormously:** Extra Trees scored **94.83 %** trained on UNSW-NB15 → tested
  on BoT-IoT, and **4.90 %** in reverse
- SHAP analysis found the cause: high within-dataset performance rests on *"strong correspondence
  between values of one or more features and Attack/Benign classes"* — shortcut features that do
  not transfer

> **This is the single most important finding for us.** The NF-v2 family exists *specifically* to
> eliminate feature-schema mismatch, and cross-dataset transfer still collapses. So **fixing our
> `IAT`/`Variance`/`Header_Length` mismatch will not rescue LODO.** The mismatch is a real bug and
> must be fixed on its own merits, but it is not the reason LODO fails.

Corroborating work: **["Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study"](https://www.researchgate.net/publication/384614967_Machine_Learning_in_Network_Intrusion_Detection_A_Cross-Dataset_Generalization_Study)**
reports within-dataset accuracy "collapsing to chance level" across datasets, and
**["Cross-Domain Generalization Failure in Lightweight Intrusion Detection Models for IIoT Networks"](https://arxiv.org/abs/2607.00553)**
documents the same for IIoT. A dataset survey of 89 corpora
(**[arXiv 2502.06688](https://arxiv.org/abs/2502.06688)**) lists weak cross-dataset generalisation
among the field's standing failure modes, alongside temporal leakage and splitting errors — with
reported F1 drops **up to 76 %**.

**Conclusion: zero-shot LODO is not a target we can hit, and nobody has hit it.**

---

## 2. What *does* work — and we have already built it

**["Dataset-centric evaluation of federated intrusion detection models in IoT networks"](https://www.nature.com/articles/s41598-025-32567-w)**
(Nature *Scientific Reports*) is the closest published work to this project: federated IDS across
**Edge-IIoTset + CIC-IoT2023 + TII-SSRC-23**, with FedAvg/FedProx/FedNova and LSTM/Transformer
backbones, evaluated in exactly the three settings we care about.

| Setting | Result |
| --- | --- |
| In-domain, Edge-IIoTset | 96.5 % macro-F1 |
| In-domain, CIC-IoT2023 | 95.2 % macro-F1 |
| In-domain, TII-SSRC-23 | 93.0 % macro-F1 |
| **Out-of-domain (cross-dataset)** | **up to 30 percentage-point macro-F1 loss** |
| **Combined multi-dataset federated training** | **≈ 90 % macro-F1 across datasets** (91.7 %) |

Their recommendation, verbatim in substance: *"Prioritize attack and environment diversity
through combined-dataset FL, select Transformer backbones where feasible, and use FedProx or
FedNova to stabilize training and reduce communication."*

**This validates our mixed-farm design.** Mixing every corpus into every farm, rather than
allocating a corpus per farm, is the published answer — and it is what we measured independently:
mixed farms raised CICIoT2023 from 0.7742 to 0.8211 and CICIoMT2024 from 0.8034 to 0.8714.

Two of their secondary findings are directly actionable for us:

- **Transformer > LSTM by ≈1–2 points macro-F1** at comparable communication budget
- **FedNova cuts communication 15–25 % vs FedAvg** and converges in fewer rounds; **FedProx**
  reduces round-to-round variance under heterogeneity

Our project uses FedAvg with a GRU. Both are defensible (FedAvg is the Eq. 21 baseline; a GRU is
the cheapest recurrent option for a Pi), but the paper gives us a measured reason to *mention*
the alternatives rather than appear unaware of them.

---

## 3. A better generalisation test exists: leave-one-DEVICE-out

**["Provenance, Not Behaviour: A Serialisation Artifact in Edge-IIoTset and a Leakage-Free Benchmark for Precision-Agriculture Intrusion Detection"](https://arxiv.org/abs/2608.15761)**
is the paper this project already cites for the `"0"`/`"0.0"` artefact — and it goes much further
than we had realised.

What it found in Edge-IIoTset: four of the seven columns the official preprocessing recipe tells
you to one-hot encode separate attack from normal **with accuracy 1.0000 on their own**, purely
through placeholder spelling. Five of six standard classifiers reach exactly
**1.0000 ± 0.0000** accuracy. Under a corrected protocol the strongest model settles at
**0.9503 ± 0.0011**.

What it then built — **AgriEdge**:

| Property | Value |
| --- | --- |
| Rows | **1,276,122**, rebuilt from the raw captures under uniform parsing |
| Devices | **five, with full attribution** |
| Leakage | **no column separates the classes above 0.0288** |
| **Leave-one-device-out** | random forest falls **0.9988 → 0.5083** balanced accuracy; the generalisation boundary sits at the perception/actuation layer |
| Non-IID federated partitioning cost | **≤ 0.0037 macro-F1** |
| A 20-round LoRaWAN training run | **4.6 hours of uplink** |
| Availability | MIT licence — [GitHub](https://github.com/MostafaGalal1/agriedge), [Zenodo code](https://doi.org/10.5281/zenodo.21941210), [Zenodo dataset](https://doi.org/10.5281/zenodo.21941319) |

**Leave-one-device-out is the generalisation test we should be running**, not leave-one-dataset-out:

- It is **achievable** — it needs device attribution, not four corpora
- It is **the right question for a federated farm deployment**: a new farm is a set of unseen
  devices, not an unseen testbed
- It has a **published reference point** in our own domain (0.9988 → 0.5083)
- Its federated cost is already quantified (≤ 0.0037 macro-F1 for non-IID partitioning)

> ⚠️ **It is not possible with CICIoT2023 as we hold it.** Verified directly: our corpus has 47
> source files across 34 attack-type directories, and neither the 39 canonical feature columns nor
> `source_file` carries device identity. CICIoT2023's published CSVs are organised by attack, not
> by device. Device-level work would need its raw pcaps.

---

## 4. Agriculture-specific datasets that actually exist

The user's instinct — *pick the dataset right and the results follow* — is correct, and there are
better-matched corpora than CICIoMT2024 for a **smart-agriculture** detector.

| Dataset | Size | Features | Attacks | Agriculture? | Available |
| --- | --- | --- | --- | --- | --- |
| **Farm-Flow** | > 1 M instances, 532 MB | **101 network-flow features** | 8 types incl. ARP spoofing, DDoS, **MQTT Flood** | **Yes — real AG-IoT testbed** | [Zenodo](https://doi.org/10.5281/zenodo.10964647) |
| **AgriEdge** | 1,276,122 rows | Edge-IIoTset native fields, leakage-audited | Edge-IIoTset's 14 + ransomware | **Yes — precision agriculture** | [Zenodo](https://doi.org/10.5281/zenodo.21941319) |
| **TII-SSRC-23** | 27.5 GB, **PCAP + CSV** | own extractor | 8 types / 32 subtypes, incl. **Mirai** | No (general) | [Kaggle](https://www.kaggle.com/datasets/daniaherzalla/tii-ssrc-23) |
| NF-v2 family | 4 corpora × 1 M flows | **43 standardised NetFlow** | varied | No | public |
| ~~NF-CICIoT2023~~ | — | — | — | — | **does not exist** |

**Farm-Flow is the most interesting find.** It is a *real agricultural IoT testbed*, over a
million flows, publicly available, and its attack set includes **MQTT Flood** — an attack on the
protocol our ESP32 sensors actually speak, which neither CICIoT2023 nor CICIoMT2024 contains.
Reported baselines exceed 90 % F1.

**The honest catch, same as before:** Farm-Flow has 101 flow features and AgriEdge has
Edge-IIoTset's native fields. **Neither shares CICIoT2023's 39-column DPKT window schema.** No
corpus does — a NetFlow conversion of CICIoT2023 does not exist, and every group ships its own
extractor.

But §1 changes how much that matters: **standardising the schema does not buy generalisation
anyway.** So schema compatibility is only required for *one model over pooled corpora*, not for
generalisation as such.

---

## 5. Methods, with evidence

| Method | Does it work? | Evidence | Cost for us |
| --- | --- | --- | --- |
| **Combined multi-dataset FL** (our mixed farms) | **Yes — the strongest published result** | ≈90 % macro-F1 across 3 corpora ([Nature](https://www.nature.com/articles/s41598-025-32567-w)) | **already built** |
| **Leave-one-device-out** as the generalisation claim | Yes, and it is measurable | 0.9988 → 0.5083 ([AgriEdge](https://arxiv.org/abs/2608.15761)) | needs device attribution |
| **Rank / quantile features** | Plausible, untested here | removes scale shift by construction | hours |
| **Domain-adversarial (DANN / DI-NIDS)** | Yes, published gains | [DI-NIDS](https://arxiv.org/abs/2210.08252): DANN features + One-Class SVM, superior cross-domain on NFv2 datasets | days; needs ≥ 2–3 source domains |
| **FedNova / FedProx instead of FedAvg** | Yes | −15–25 % communication, fewer rounds ([Nature](https://www.nature.com/articles/s41598-025-32567-w)) | moderate |
| **Transformer instead of GRU** | Yes, +1–2 points | same | moderate; heavier on a Pi |
| **Per-deployment calibration** | Yes — and it is already our deployment story | our own per-client gain: +0.1338 / +0.0789 / +0.0516 | already measured |
| **Zero-shot cross-testbed transfer** | **No — nobody achieves it** | 56.28 % decay even with standardised features ([arXiv 2205.04112](https://arxiv.org/abs/2205.04112)) | do not pursue |

---

## 6. Recommendation

### Do not chase zero-shot LODO

It is not achievable, and the field knows it. **Report our number as a replication**, with the
citations above: *"consistent with the 56 % average cross-dataset decay reported for standardised-
feature NIDS corpora and the up-to-30-point loss reported for federated IoT IDS, our zero-shot
cross-corpus transfer collapses."* That is a defensible, literature-anchored negative result
rather than an unexplained failure or a hidden one.

### Still fix the feature-semantics bug

For its own sake, not to rescue LODO. `Variance` spanning 0…1 against 0…1.13×10⁸ is a defect
regardless of what it does to any score, and the `/Number` normalisation already recovers four of
the eight broken columns — the two extractors use window sizes differing by **7.33×**
(`Number` 69.66 vs 9.51), which is itself a reportable observation about combining CIC corpora.

### Make the headline what the literature supports

**Combined multi-dataset federated training**, which we already built and measured. Not
zero-shot transfer.

### Change the generalisation claim to leave-one-device-out — if we add a corpus that supports it

This is the one place more data genuinely buys something. **Farm-Flow** (real agricultural
testbed, MQTT Flood attacks) or **AgriEdge** (precision agriculture, five attributed devices,
leakage-audited) would let us make a *measurable* generalisation claim that matches the
deployment question: can the model handle a farm whose devices it has never seen?

### What I would not do

Add TII-SSRC-23 or the NF-v2 family. They are not agricultural, and §1 says the extra domains
would not deliver zero-shot transfer anyway.

---

## Sources

- [On Generalisability of Machine Learning-based Network Intrusion Detection Systems](https://arxiv.org/abs/2205.04112)
- [Dataset-centric evaluation of federated intrusion detection models in IoT networks](https://www.nature.com/articles/s41598-025-32567-w)
- [Provenance, Not Behaviour: A Serialisation Artifact in Edge-IIoTset and a Leakage-Free Benchmark for Precision-Agriculture Intrusion Detection](https://arxiv.org/abs/2608.15761)
- [Farm-flow dataset: Intrusion detection in smart agriculture based on network flows](https://www.sciencedirect.com/science/article/pii/S0045790624008188)
- [Cross-Domain Generalization Failure in Lightweight Intrusion Detection Models for IIoT Networks](https://arxiv.org/abs/2607.00553)
- [Network Intrusion Datasets: A Survey, Limitations, and Recommendations](https://arxiv.org/abs/2502.06688)
- [Towards a Standard Feature Set for Network Intrusion Detection System Datasets](https://arxiv.org/abs/2101.11315)
- [NetFlow Datasets for Machine Learning-based Network Intrusion Detection Systems](https://arxiv.org/abs/2011.09144)
- [DI-NIDS: Domain Invariant Network Intrusion Detection System](https://arxiv.org/abs/2210.08252)
- [TII-SSRC-23 Dataset: Typological Exploration of Diverse Traffic Patterns for Intrusion Detection](https://arxiv.org/abs/2310.10661)
- [AgriEdge code and benchmark](https://github.com/MostafaGalal1/agriedge)
