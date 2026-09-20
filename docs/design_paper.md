## Domain Analysis, Literature Review, and System Federated GRU Intrusion Detection with Ascon-Authenticated Telemetry for Smart Agriculture IoT: Design

Abhishek Jadli

Aryan Agarwal

Advait Amit Malviya Department of Computer

Department of Computer

Department of Computer

Science

Science

Science

Vellore Institute of Technology

Vellore Institute of Technology

Vellore Institute of Technology

Chennai, India

Chennai, India

Chennai, India

abhishek.jadli2023@vitstudent.ac.in

aryan.agarwal2023@vitstudent.ac.in

advaitamit.malviya2023@vitstudent.ac.in

Dr. Vijayprabhakaran K School of Computer Science and

Engineering

Vellore Institute of Technology

Chennai, India

Abstract—Smart farms run on sensor readings that trigger physical actions: a moisture value decides whether a pump starts. Two security requirements follow, and they arrive together rather than in sequence. An edge gateway has to recognise attack traffic from compromised devices locally, because rural backhaul cannot carry every packet to a cloud inspection service. It also has to protect the readings that survive inspection, since a modified value that arrives silently does more damage to a farm than one that is merely observed. The literature treats these as separate problems. This paper presents the domain analysis, literature and patent review, and system design for an architecture that treats them as one. Three simulated edge nodes train a gated recurrent unit (GRU) intrusion detector on the CICIoT2023 benchmark using sample-weighted federated averaging, so raw traffic records stay on the node that captured them. The detector’s verdict then selects the data path. Telemetry judged benign is protected with Ascon-AEAD128, the authenticated encryption scheme standard- ised in NIST SP 800-232; telemetry judged malicious is diverted to an alerting path and never reaches the cloud channel. We describe the six-parameter protocol used to select the twenty studies reviewed, set out the six gaps it exposes, and specify the detection, federation and cryptographic layers formally. Two of the controls we specify, a deduplication-ordering leakage control and a runtime feature-provenance boundary, are left unstated in comparable work. Under our parameterisation federation costs roughly 0.77 MiB per round, against about 276 MB to pool the same data centrally.

Index Terms—Federated learning, intrusion detection, gated recurrent unit, lightweight cryptography, Ascon, authenticated encryption, smart agriculture, Internet of Things, CICIoT2023.

## I. DOMAIN AND PROBLEM DEFINITION

## A. The Deployment Environment

Smart agriculture places networked sensing and actuation across farmland. A typical deployment contains soil-moisture probes, temperature and humidity sensors, water-level and flow meters, weather stations, and programmable irrigation controllers. These devices publish readings over lightweight application protocols, most commonly MQTT, to a farm-side gateway, which forwards them to a cloud tier that decides when to irrigate or fertilise.

This is a cyber-physical system, not a data-collection net- work, and the distinction matters for everything that follows. A corrupted soil-moisture reading does not produce a misleading chart. It causes a pump to run, or to stay off when it should not. The damage is physical, it appears some days later, and it is expensive: lost crop, wasted water, fertiliser burn, equipment damage. Integrity of the data therefore matters at least as much as confidentiality, and both matter more here than they would in a consumer deployment.

Four properties of the field environment shape every design decision that follows.

C1—Resource constraints. Field nodes are battery- or solar-powered microcontrollers whose energy budget for cryp- tography and inference is measured in microjoules. This is the class of device that motivated the NIST lightweight cryptography programme.


*TABLE I*

*ATTACK FAMILIES IN CICIOT2023 AND THEIR AGRICULTURAL*

*CONSEQUENCE.*

| Family | Consequence in a farm deployment |
| --- | --- |
| DDoS / DoS | Telemetry never arrives; controllers act on stale |
|   | state |
| Mirai | Nodes conscripted into a botnet; bandwidth |
|   | exhausted |
| Reconnaissance | Deployment mapped; precursor to targeted |
|   | compromise |
| Spoofing | False readings injected; wrong irrigation decisions |
| Brute force | Default-credential takeover of gateways |
| Web-based | Compromise of the farm management interface |

C2—Constrained backhaul. Rural connectivity is usually cellular (NB-IoT, LTE-M) or long-range radio (LoRaWAN), with metered bandwidth and frequent outages. Continuously shipping raw captures to a central service for training is not bandwidth-feasible, a claim we quantify in Section III-F3. [URL 🔗](#page-0)

C3—Physical exposure. Nodes sit in open fields, where an adversary can approach them or capture radio traffic with commodity hardware. The assumption of a trusted internal network, reasonable inside a building, does not hold.

C4—Data ownership. Farm operational data is commer- cially sensitive because it predicts yield. Cooperatives are increasingly unwilling to hand raw records to a third-party aggregator, a reluctance reinforced by regimes such as India’s DPDP Act, 2023 and the EU GDPR.

## B. Threat Model

We distinguish three channels, each with a different ad- versary, because conflating them is a common source of overstated security claims.

Channel 1: device-to-gateway network traffic. The ad-

versary is a compromised IoT device on the same network. This is the scenario the CICIoT2023 benchmark was built to represent, in which all malicious traffic originates from com- promised IoT devices attacking other IoT devices. Relevant attack families and their agricultural consequences are given in Table I. The defence is detection. [URL 🔗](#page-0)

Channel 2: gateway-to-cloud telemetry. The adversary is a network observer who can read, modify, replay, or drop messages in transit. Detection does not help here, because the traffic has already been judged benign. The defence is authenticated encryption.

Channel 3: node-to-aggregator model updates. The ad- versary is either a curious aggregator that would like to reconstruct client data, or a compromised client submitting a poisoned update. Federated learning addresses the first only partially and the second not at all. Section I-F returns to both. [URL 🔗](#page-0)

## C. Why the Three Sub-Problems Are Coupled

Most existing work solves one of the three problems below. Our position is that a deployable architecture needs all three, and that composing them raises engineering problems that none of them raises alone.

Detection must run at the edge. Constraint C2 rules out backhauling traffic for cloud inspection, and there is a second reason besides bandwidth. Flooding and scanning attacks are defined by how traffic changes over a short interval: its rate, its burst structure, the gaps between arrivals. A classifier that sees one record at a time discards that information before it begins. We therefore use a recurrent model.

## Learning must be collaborative but data must stay local.

A single farm sees too few attack instances to train a classifier that generalises, and the rarer families may never appear on its network at all. Pooling records centrally fixes the statistics but breaks C4 and C2 at once. Federated learning exchanges parameters instead of records, which is the compromise we adopt.

Classification on its own protects nothing. Traffic labelled benign still has to cross an untrusted network. Confidentiality is not sufficient here: a scheme without integrity hands the receiver a modified moisture reading that is indistinguishable from a genuine one, and the farm then acts on it. What the channel needs is authenticated encryption, so that tampering produces a rejection.

## D. Formal Problem Statement

Let D = {(xt, yt)}M t=1 denote a corpus of network flow records, where xt ∈ RF0 is a vector of flow-derived features and yt ∈ C is a traffic-class label drawn from a taxonomy C with |C| = C. After feature selection to F ≤ F0 dimensions and windowing with length W, a training example is a matrix

We seek a classifier fθ : RW×F → ∆C−1 mapping a window to a probability simplex, together with a deterministic binary projection

The parameters θ are to be obtained without centralising D. With K clients holding disjoint partitions D1, . . . , DK of

sizes n1, . . . , nK and n = P nk k nk, the learning objective is

subject to the constraint that no element of Dk is transmitted beyond client k.

Finally, let p be an application-layer telemetry payload associated with the window and a its associated metadata. The system must implement the routing rule

where Enc is an authenticated encryption function under an edge key ke, and the two output paths Πcloud and Πalert


*TABLE II*

*ASSUMPTIONS, CLASSIFIED BY STATUS.*

| ID Assumption | Status |
| --- | --- |
| A1 Three edge nodes are simulated | Simulation; independence |
| as independent processes on one | enforced in code |
| machine |   |
| A2 Partition-to-farm-region mapping | Simulation; documented as |
| is assigned, not intrinsic to the | a strategy |
| data |   |
| A3 Runtime network features come | Simulation; declared |
| from held-out records, not from | limitation |
| the JSON payload |   |
| A4 The aggregator is | Scope boundary; not solved |
| honest-but-curious |   |
| A5 Ascon keys are pre-shared out of | Prototype; production key |
| band | management out of scope |
| A6 The cloud tier is a mock receiver Scope boundary |   |
| A7 The class taxonomy is known to | Design requirement for a |
| all clients a priori | shared output layer |

are disjoint. Equation (5) states the coupling formally. The classifier output is consumed as a control decision over the data path rather than written to a log. [URL 🔗](#page-0)

## E. Assumptions and Their Status

Table II lists the assumptions the design rests on. We sepa- rate them by status, because some are properties of the world and others are artefacts of working in simulation. Presenting a simulation artefact as an empirical fact is a common failure in this class of system, and A3 is the assumption comparable work most often leaves unstated. Section III-H addresses it directly. [URL 🔗](#page-0)

## F. Objectives, Scope, and Expected Outcomes

1) Objectives: The work has five objectives, each stated with a criterion by which it can be judged complete.

- O1 Characterise CICIoT2023 empirically and build a prepro- cessing pipeline that is provably free of train–test leakage. Criterion: deduplication precedes splitting, all transforms are fitted on training partitions only, and both properties are asserted by automated tests.

- O2 Train a sequence GRU detector and establish a centralised performance reference. Criterion: macro-F1 over eight classes with per-class breakdown, confusion matrix, and binary false-positive rate.

- O3 Implement K = 3 logically independent clients with a documented non-IID partition and sample-weighted FedAvg. Criterion: the global model loads into every client, and per-client class histograms are published.

- O4 Quantify the cost of federation. Criterion: a three-way comparison of centralised, federated-global, and local- only models on a common held-out test set, over at least three seeds.

- O5 Integrate Ascon-AEAD128 on the benign path and demonstrate rejection of both single-bit ciphertext and single-bit associated-data mutations.

2) Scope: In scope: dataset characterisation, leakage- controlled preprocessing, feature selection, sequence construc- tion, the GRU detector with binary projection, three-client federated simulation, MQTT/JSON telemetry simulation with an explicit feature adapter, the Ascon benign path, the alerting interface, a mock cloud receiver, and the evaluation protocol of Section III-I. [URL 🔗](#page-0)

Out of scope, and reported as limitations rather than as solved problems: Byzantine-robust aggregation, differen- tial privacy, secure aggregation, gradient-inversion defences, adversarial-example robustness, production key management and device attestation, dashboards, cloud agronomic analytics, and physical hardware deployment.

3) Expected outcomes: A reproducible dataset and feature report; a trained detector reported with macro-F1 and false- positive rate rather than accuracy alone; a quantified answer to the question of how much detection performance is sacrificed by keeping data local; a working authenticated channel with measured latency and byte overhead; and a tested, single- command reproducible implementation.

## G. Contributions

This paper contributes: (i) a threat analysis separating three channels, showing why detection and payload protection must be co-designed (Section I-B); (ii) a six-parameter selection protocol applied to twenty studies, with the gap analysis it yields (Section II); and (iii) a formal system design covering the detection, federation, and cryptographic layers, including two controls comparable work omits: a deduplication-ordering leakage control and an explicit feature-provenance boundary (Section III). We claim no new algorithm; the contribution is the composition and the rigour of its specification. [URL 🔗](#page-0)

## II. LITERATURE AND PATENT REVIEW

## A. Review Protocol

The review answers three questions. Which detection archi- tectures have been shown to work on IoT flow data? What does federated training cost in that setting? And has any prior system connected a detection verdict to a cryptographic data path? Sources were drawn from IEEE Xplore, the ACM Digital Library, SpringerLink, MDPI, ScienceDirect, arXiv, the NIST Computer Security Resource Center, and Google Patents.

Search strings combined four term groups: ("federated

learning" OR FedAvg),

detection" OR IDS OR "anomaly detection"), (IoT OR IIoT OR "smart agriculture" OR

"precision agriculture"), and (Ascon OR "lightweight cryptography" OR AEAD OR MQTT). Records were screened by title and abstract, then by

full text against the parameters below.

("intrusion

## B. Selection Parameters

A reading list assembled by judgement is difficult to audit, so we scored every candidate against the six parameters listed in Table III. Each parameter is scored 0, 1, or 2, giving a [URL 🔗](#page-0)


*TABLE III*

*SELECTION PARAMETERS. EACH SCORED 0–2; ADMISSION AT s ≥ 8 OF*

*12.*

| Parameter | Scored on |
| --- | --- |
| P1 Domain | Whether the work targets IoT, IIoT, or agricultural |
| proximity | network security rather than enterprise or |
|   | vehicular networks |
| P2 Method overlap Use of federated optimisation, recurrent sequence |   |
|   | models, or lightweight authenticated encryption |
| P3 Benchmark | Use of CICIoT2023 or a comparable public IoT |
| comparability | flow dataset, enabling like-for-like comparison |
| P4 Evidential | Peer-reviewed venue, stated experimental protocol, |
| strength | and metrics reported beyond accuracy |
| P5 Recency | 2021 onward for applied work; foundational |
|   | exceptions admitted where the method is still the |
|   | reference implementation |
| P6 Gap-analytic | Whether the work states, or visibly exhibits, a |
| value | limitation this project can act on |

*Fig. 1. Screening funnel. Exclusion at each stage is governed by the parameters of Table III; a zero on domain proximity or method overlap is disqualifying irrespective of total score. [URL 🔗](#page-0)*

maximum of 12. A study was admitted at a threshold of s ≥ 8, with the additional rule that a score of 0 on P1 or P2 was disqualifying regardless of the total. We set the threshold there so that a paper had to be strong on both domain proximity and method overlap, rather than compensating for weakness in one with strength in the other.

Applying the protocol produced the screening funnel of Fig. 1 and a final corpus of twenty studies, S1–S20, sum- marised in Table V. Two admissions need explaining. S2 and S3 fall outside the P5 window and were admitted under the foundational exception, since federated averaging and its proximal variant are still the algorithms against which later methods are measured. S13 was admitted even though it reports a tree ensemble outperforming the recurrent models it tested. That result cuts against our own architectural choice, which is the reason we kept it in the corpus and the reason Section III-I includes a random-forest baseline. [URL 🔗](#page-0)

## C. Thread A: Benchmark Data and Sequence-Based Detection

CICIoT2023 [1] is the benchmark this work uses. It was built on a testbed of 105 real IoT devices and contains roughly 46.7 million labelled records with 46–47 flow-derived features, spanning one benign class and 33 attack classes grouped into seven families. Two of its properties drive our design. [URL 🔗](#page-0)

Each record summarises a fixed-size packet window ex- tracted from packet captures rather than a single packet. The data is therefore already aggregated once over time, and a recurrent model reading a sequence of records performs a second-order temporal analysis: it observes how flow statistics evolve. That is the justification for a recurrent architecture here, and it rests on the record semantics rather than on the general observation that network traffic is sequential.

The class distribution is severely imbalanced. Writing nc for the count of class c, the reported imbalance ratio

means that a classifier predicting the majority class alone attains high accuracy while detecting nothing of interest. Any evaluation on this dataset reporting accuracy alone is uninformative, so Section III-I treats macro-F1 and false- positive rate as the primary metrics. [URL 🔗](#page-0)

Recurrent and convolutional-recurrent architectures domi- nate the applied literature [10], [13]. S13 is the useful dissent: benchmarking nine methods, it found a random forest strongest on tabular flow features, and then compared simple, weighted, and difference-based aggregation rules federated over that model. The implication for our own choice of model is one we carry into Section III-I. [URL 🔗](#page-0)

Limitation observed across Thread A. Papers that build sequences from flow records frequently shuffle rows before windowing, or do not state the ordering procedure at all. Where the source data carries no timestamp or flow iden- tifier, a sliding window is a positional construct and not a temporal one. We found this distinction rarely acknowledged. Section III-D addresses it. [URL 🔗](#page-0)

## D. Thread B: Federated Learning for IoT Intrusion Detection

Federated averaging [2] is the reference algorithm: clients perform local stochastic gradient descent and the server aver- ages parameters weighted by local sample count. FedProx [3] adds a proximal term that constrains client drift under sta- tistical heterogeneity, and is the standard citation for why non-identically distributed client data degrades plain FedAvg. Broader context is provided by a survey of federated learning for IoT [4] and a systematic review of federated intrusion detection [5]. [URL 🔗](#page-0)

Applied results are consistent in direction. Federated deep detectors reach accuracy close to their centralised counterparts on IoT benchmarks [6], [8], including on CICIoT2023 itself, where a federated CNN-GRU ensemble under weighted aver- aging reported 98.25% accuracy [10]. Work on the industrial IoT quantifies the privacy–utility trade-off when differential privacy is added [7], and a recent aggregation rule reports that tightening the privacy budget from ε = 5 to ε = 0.5 costs only about 1.4 percentage points of accuracy [12]. Alternative architectures [9] and aggregation variants [11] fill out the space. [URL 🔗](#page-0)

S10 is the closest prior work to our detection layer. It establishes that the combination of a GRU-based model,


weighted federated averaging, and CICIoT2023 is workable, which de-risks our own detection design. It does not consider what happens to traffic after classification, does not model a runtime IoT stream, and is not set in an agricultural context.

Limitations observed across Thread B. Four recur. L1: privacy is asserted rather than bounded, with many papers treating “raw data does not move” as equivalent to privacy, which parameter-inversion attacks show it is not. L2: client partitioning is under-specified, and a uniform random split is often described as heterogeneous when it is identically distributed by construction. L3: the local-only baseline is usually absent, although it is the comparison that matters operationally, since local-only is what an operator gets by declining to federate. L4: the pipeline terminates at a label.

## E. Thread C: Lightweight Authenticated Encryption

NIST selected the Ascon family in February 2023 as the outcome of its lightweight cryptography process, and SP 800- 232 [21] specifies four primitives, of which Ascon-AEAD128 is the authenticated encryption scheme used here. The family is permutation-based over a 320-bit state, single-pass, and inverse-free; one permutation serves authenticated encryption, hashing, and extendable output, which allows a compact com- bined implementation on constrained hardware. The original design is described in [14]. [URL 🔗](#page-0)

One point of terminology has practical consequences and is worth stating plainly, because it is a common implemen- tation error. The standardised Ascon-AEAD128 is not the same parameterisation as Ascon-128 from the pre-standard Ascon v1.2; it derives from the earlier Ascon-128a, with revised initial values and a little-endian formatting change. The names were changed precisely to prevent confusion. A widely installed Python package implements the v1.2 variants only, so a system that claims conformance to SP 800-232 must verify which variant its library actually provides. Section III-G makes that verification a gating test. [URL 🔗](#page-0)

Performance evidence for Ascon on constrained platforms is substantial [15], [16], and there are direct evaluations over IoT application protocols. LWE-IoT [17] compares Ascon, PRESENT, SIMON, SPECK, and ChaCha20 over both MQTT and CoAP on ESP32 nodes across payload sizes; an Industry 4.0 framework validates Ascon over MQTT against AES- GCM [18]; and a further study integrates Ascon-AEAD128 into an MQTT exchange for robotic and IoT devices [19]. [URL 🔗](#page-0)

Limitation observed across Thread C (L5). These eval- uations benchmark the cipher on a channel that carries no intelligence about whether the traffic should be forwarded at all. Encryption and detection are treated as independent layers.

## F. Thread D: Agricultural and Application-Layer Security

Work specific to agriculture confirms both the relevance of lightweight cryptography and the shape of the deployment. A secure irrigation system for precision agriculture [20] protects published sensor data and the irrigation decisions returned to actuators, reporting improvements in power consumption, exe- cution time, and memory over AES. It uses a non-standardised [URL 🔗](#page-0)

cipher and contains no detection layer, but it establishes the threat scenario we adopt in Channel 2 of Section I-B: an adversary who alters sensor readings in transit changes what the farm does. [URL 🔗](#page-0)

## G. Patent Landscape

Four filings bound the space. US 12,301,597 [22] trains a random-forest classifier on CICIoT2023 features at the network edge using a digital-twin representation; it is the closest granted patent by dataset and placement, but training is centralised and there is no payload cryptography. An Indian filing by Vellore Institute of Technology [23] claims collab- orative training of intrusion-detection models without sharing traffic, applying differential-privacy noise to parameters before transmission and aggregating with FedAvg. That filing is adjacent to our federation layer, and it adds the differential privacy we place out of scope. A further filing claims privacy- preserving global threat detection with standardised threat- intelligence dissemination [24], and a multi-level federated framework for Industry 4.0 [25] establishes prior art for federation across heterogeneous device tiers. [URL 🔗](#page-0)

The conclusion here is a narrow one. Federated intrusion de- tection is actively claimed, and edge detection on CICIoT2023 is actively claimed. What the surveyed landscape does not contain is an architecture in which a standardised lightweight- AEAD data path is conditioned on the output of a federated detector. What appears unclaimed is the composition, while the individual components are all well covered.

## H. Gap Analysis and Positioning

Table IV lists the six gaps the protocol exposed and the design response to each. Two need further comment. [URL 🔗](#page-0)

G1 is the structural gap and the reason for the work. Thread B terminates at a label; Thread C benchmarks ciphers on a channel that carries no verdict. Neither literature asks what the detector’s output should do next. Our answer is Equation (5): the verdict selects the data path, and the two paths are kept disjoint in the code rather than by convention. [URL 🔗](#page-0)

G6 arises only when a trained detector is wired to a simulated device stream, which is why the surveyed work does not run into it. A model trained on flow-derived fea- tures cannot consume an application-layer JSON payload. A demonstration that pretends otherwise establishes nothing at all, so Section III-H defines the boundary explicitly and says what the resulting demonstration does and does not show. [URL 🔗](#page-0)

## III. PROPOSED METHODOLOGY AND SYSTEM DESIGN

## A. Architectural Overview

The system has two planes that share a model but nothing else. Fig. 2 shows both. The training plane runs offline and periodically: clients train locally and exchange parameters with an aggregator. The runtime plane runs continuously: telemetry arrives, is classified using the current global model, and is routed according to Equation (5). [URL 🔗](#page-0)

The separation between the planes is enforced in the im- plementation and not only in the figure. They carry different


*TABLE IV*

*GAPS IDENTIFIED BY THE REVIEW PROTOCOL, AND THE DESIGN*

*RESPONSE.*

| Gap | Response |
| --- | --- |
| G1 Detection and payload | Verdict governs the data path; paths |
| protection studied separately | disjoint by construction, Eq. (5) |
| (L4, L5) |   |
| G2 Federated privacy asserted, | Claim restricted to “raw records are |
| not bounded (L1) | not transmitted”; inversion and |
|   | poisoning named as unaddressed |
| G3 Client partitions | Dirichlet partition with α as an |
| under-specified and usually IID | experimental variable; histograms |
| (L2) | published |
| G4 Local-only baseline absent | Three-way comparison is a primary |
| (L3) | result, Section III-I |
| G5 Sequence ordering | Ordering, homogeneity, and labelling |
| unjustified (Thread A) | rules stated and defended, |
|   | Section III-D |
| G6 Runtime feature provenance | Explicit adapter boundary with |
| unaddressed | declared limitation, Section III-H |

payloads, to different endpoints, under different trust assump- tions, and the failure mode we are guarding against is a code path that allows a malicious-verdict payload to reach the cloud transport. In the implementation the two planes share only an abstract transport interface; the malicious handler holds no reference to the cloud client at all, and an automated test asserts that no encrypted payload is emitted during a malicious-only run.

## B. Data Preparation and Leakage Control

Before any modelling, a characterisation script reports the actual column names and types, per-column null and zero- variance checks, the exact duplicate count, the label vocabulary with counts, and the numeric correlation matrix. No column or class name enters the final report unless this script produced it. Figures quoted from the literature are treated as hypotheses to confirm.

1) Subsampling: The full corpus does not fit in workstation memory. We draw a stratified, capped subsample of M ≈

1.5–2

order and applying a per-class cap

chosen to compress the dominant DDoS classes while retain- ing every available instance of the rare families. The seed and the resulting counts are recorded in a run manifest.

2) Deduplication ordering: CICIoT2023 contains exact du- plicate rows. This creates a leakage channel that is easy to miss. If deduplication is applied after the train–test split, iden- tical records appear on both sides, and the test set measures memorisation rather than generalisation. Writing δ for the duplicate fraction and ρ for the probability that a duplicate’s twin lands in the training partition, the fraction of test items that are effectively seen during training is approximately δρ, and the observed accuracy is inflated by roughly

× 106 records by reading whole CSV parts in fixed

κc,

The inflation is largest for a genuinely difficult class, where Acctrue is low, so a detector is flattered most on exactly the classes where its performance matters. We therefore fix the order as

and assert it with a test that fails if any record hash appears in both partitions.

3) Taxonomy and splitting: The primary task is C = 8 classes: benign plus seven attack families. At IR ≈ 5751, the rarest of the 33 leaf classes cannot support a defensible macro- F1 estimate, and the family label is in any case the actionable granularity for an operator. The binary decision of Equation (5) is a deterministic projection of the eight-class output; we do not train a second model for it. [URL 🔗](#page-0)

Splitting operates on contiguous blocks of records rather than individual rows, so that local ordering survives into sequence construction. A stratified subset of blocks forms a global test set shared by every client and every baseline; the remainder is partitioned across clients, and each client splits its own blocks into local training and validation sets.

## C. Feature Selection

Subsampling reduces rows and feature selection reduces columns. The two are occasionally conflated in the literature, so we keep them separate here. Our selector runs in four stages, all fitted on training blocks only, and is persisted so that inference uses exactly the same columns.

Stage 1: rule-based elimination. Drop labels, identifiers, zero-variance columns, and any column that could not exist at inference time.

Stage 2: correlation pruning. On training data, compute the Spearman rank correlation ρjk for every numeric pair and remove one member of any pair with |ρjk| ≥ τ, default τ = 0.95, retaining the member with higher univariate rel- evance.

Stage 3: relevance ranking. Score each surviving feature j by mutual information with the label,

and independently by random-forest impurity importance. The two rankings are combined by reciprocal rank fusion,

where rm(j) is the rank of feature j under method m. Two methods are used because they disagree in informative ways: mutual information captures non-monotonic univariate dependence, while impurity importance is biased toward high- cardinality features. Where they disagree we report the dis- agreement instead of hiding it in an average.

Stage 4: cardinality selection. Sweep F ∈ {8, 12, 16, 24, F0} and choose the knee of the validation macro-F1 curve, balancing performance against inference cost and interpretability.

D. Temporal Sequence Construction

This section addresses gap G5, and the resolution is a stated assumption rather than a technique.

The distributed CSV files are not expected to carry a timestamp or a flow identifier. Without one, per-device chronological ordering cannot be recovered, and a sliding window over rows is positional rather than temporal. Much of the surveyed work slides a window anyway, which claims more than the data supports.

We construct windows only over contiguous same-label runs within a single source file, preserving whatever ordering the capture-to-CSV extraction imposed, and we record this as an assumption instead of presenting it as recovered chronology.

Rows are never shuffled before windowing; shuffling is applied to complete sequences when batches are formed. Three rules follow. Training windows do not cross a label boundary, so supervision stays unambiguous. The sequence label is that of the final record, (y_i = y_{i+W-1}), which matches the online semantics of deciding what is happening now given recent history. A separate held-out set of mixed windows spanning a benign-to-attack boundary is reserved for evaluation only, and is used to measure detection latency: how many records into an attack the verdict takes to flip.

Given runs of lengths ({L_r}), the number of sequences is

[
N_{\mathrm{seq}} = \sum_r \max\left(0, L_r - W + 1\right).
\tag{12}
]

which makes explicit that short runs contribute nothing once (W) grows large. That cost of increasing the window is easy to overlook. We sweep (W \in {1, 8, 16, 32}), and include (W = 1) as an ablation. If it matches (W = 16), recurrence has not earned its place in the architecture, and we will say so.

## E. Detection Model

Each client holds an identical architecture, which is a precondition for parameter averaging. A single GRU layer reads the window and a linear head produces class logits. For input (x_t \in \mathbb{R}^{F}) and hidden state (h_{t-1} \in \mathbb{R}^{H}),

[
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z),
\tag{13}
]

[
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r),
\tag{14}
]

[
\tilde{h}t =
\tanh\left(W_h x_t + U_h(r_t \odot h{t-1}) + b_h\right),
\tag{15}
]

[
h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t,
\tag{16}
]

with (\sigma) the logistic function and (\odot) the elementwise product.

The update gate (z_t) is what lets the model carry evidence across the window: a burst that begins early remains represented in (h_W).

Class scores follow from the final state,

[
p = \operatorname{softmax}\left(W_o,LN(h_W) + b_o\right),
\tag{17}
]

where (LN) denotes layer normalisation. The choice of layer normalisation over batch normalisation is deliberate and specific to the federated setting: batch statistics are running estimates of a client’s own input distribution, so averaging them across clients with different distributions produces a quantity that describes no client’s data. Layer normalisation has no such cross-client state.

Training minimises class-weighted cross-entropy,

-\sum_{c=1}^{C} w_c I[y=c]\log p_c,
\qquad
w_c = \frac{n}{C n_c},
\tag{18}
]

with weights computed on training data only. Equation (18) is the first line of defence against the imbalance of Section II-C.

We do not apply synthetic oversampling, because interpolating between flow records fabricates temporal structure that never occurred.

The parameter count is

\underbrace{3(FH + H^2 + 2H)}{\mathrm{GRU}}
+
\underbrace{2H}{LN}
+
\underbrace{HC+C}_{\mathrm{head}}.
\tag{19}
]

With (F = 16), (H = 96), and (C = 8) this gives (32{,}832 + 192 + 776 = 33{,}800) parameters, or 132 KiB at single precision. That is small enough to be plausible on an edge gateway and, as the next section shows, small enough that the communication cost of federation is not the binding constraint.

## F. Federated Training Design

1) Partitioning

To avoid the under-specification identified as L2, clients are formed by a Dirichlet partition applied at block level. For each class (c) we draw

[
p_c \sim \operatorname{Dir}(\alpha \mathbf{1}_K).
\tag{20}
]

and allocate class-(c) blocks to the (K) clients in those proportions. Small (\alpha) yields strong heterogeneity; large (\alpha) approaches an identical distribution. We treat (\alpha) as an experimental variable, which lets us measure sensitivity to heterogeneity instead of assuming a level of it, and we report per-client class histograms so the partition can be inspected. Partition is applied to blocks rather than rows so that temporal locality survives into each client’s sequence construction.

2) Aggregation

Each round (r), every client initialises from the global parameters, performs (E) local epochs, and returns (\theta_k^{(r)}). The server computes

\sum_{k=1}^{K}
\frac{n_k}{n}\theta_k^{(r)}.
\tag{21}
]

Here (n_k) is the number of training sequences held by client (k), not the number of raw rows. The sequence is the training example, so it is the correct weight; using row counts would systematically over-weight clients whose data happens to be fragmented into short runs, since those clients yield fewer sequences per row by Equation (12). Unweighted averaging is retained as an ablation.

Optimiser state is local and is not aggregated: a fresh optimiser is constructed each round. Averaging momentum buffers is a different algorithm, and performing it silently would make the reported method not FedAvg. Parameters are serialised in a structured tensor format rather than by pickling, since arbitrary-object deserialisation across a transport boundary is a code-execution hazard even in a prototype. The procedure is given as Algorithm 1.

Algorithm 1 — One federated round

Require: global (\theta^{(r)}), clients (1..K), local epochs (E)

for each client (k) in parallel do

(\theta_k \leftarrow \theta^{(r)})

construct a fresh optimiser

for (e = 1) to (E) do

for each minibatch ((X,y) \subset D_k) do

(\theta_k \leftarrow \theta_k - \eta\nabla\ell(f_{\theta_k}(X),y))

end for

end for

send ((\theta_k,n_k)) {no record of (D_k) is sent}

end for

(\theta^{(r+1)} \leftarrow \sum_k (n_k/n)\theta_k)

broadcast (\theta^{(r+1)}) to all clients

3) Communication cost

Per round, each client uploads and downloads one parameter vector, so the total traffic is

[
B_{\mathrm{round}} = 2K|\theta|b,
\tag{22}
]

with (b = 4) bytes at single precision. For (K = 3) and (|\theta| = 33{,}800) this is 811 kB, about 0.77 MiB, and a 20-round schedule costs roughly 15.5 MiB in total.

The centralised alternative is to move the records themselves: at (5\times10^5) records per client and 46 features, that is about 92 MB per client and 276 MB overall, a factor of roughly 18 larger. This is the quantitative form of the argument in constraint C2, and it holds because the model is small. It would not hold for a large model retrained frequently, which is a reason to prefer a compact detector independently of edge inference cost.

4) Federated preprocessing statistics

A subtlety arises that FedAvg descriptions usually skip. If each client fits its own scaler, the global model sees three different input distributions and evaluation on a shared test set is ill-defined. If one scaler is fitted on pooled raw data, the no-raw-data constraint is violated.

We resolve this by having clients transmit only sufficient statistics, namely count, sum, and sum of squares per feature, which the server combines exactly using the parallel formulation

[
n = \sum_k n_k,
\qquad
\mu = \frac{1}{n}\sum_k n_k\mu_k,
\tag{23}
]

[
M_2 =
\sum_k
\left(
M_{2,k} + n_k(\mu_k-\mu)^2
\right),
\qquad
\sigma^2 = \frac{M_2}{n}.
\tag{24}
]

The result is identical to a scaler fitted on the pooled training data, but only aggregates cross the boundary. The global scaler is distributed once, before round one.

## G. Authenticated Encryption Design

The benign path uses Ascon-AEAD128 as specified in NIST SP 800-232, with a 128-bit key, 128-bit nonce, and 128-bit tag.

Encryption is

[
(c,t) = \operatorname{Enc}_{k_e}(\nu,a,p),
\tag{25}
]

and decryption returns the payload only if the tag verifies:

[
\operatorname{Dec}_{k_e}(\nu,a,c,t) \in {p,\perp}.
\tag{26}
]

The (\perp) outcome is the property that matters for this application. A scheme offering confidentiality alone would return some plaintext for a modified ciphertext, and the farm would act on it.

1) Library conformance

Given the naming issue described in Section II-E, the implementation is verified against the official known-answer test vectors as a gating unit test before any integration work, and the package, version, and variant string are recorded in the run manifest. No primitive is implemented from scratch and no placeholder encryption is used at any stage.

2) Associated data

The associated data is authenticated but not encrypted. The cloud receiver must read the edge identifier before it can select a decryption key, so the identifier has to sit outside the ciphertext. We therefore set

[
a = \langle edge_id, device_id, counter, schema_version\rangle.
\tag{27}
]

Authenticating a monotonic counter also gives replay detection, which encryption alone would not provide.

3) Nonce discipline

Nonce reuse under a fixed key is catastrophic for AEAD security, so a fresh 16-byte nonce is drawn per message from a cryptographically secure generator and transmitted alongside the ciphertext.

For (q) messages under one key, the birthday bound gives

[
\Pr[\mathrm{collision}]
\le
\frac{q(q-1)}{2^{129}}.
\tag{28}
]

which for (q = 10^6) is below (1.5\times10^{-27}): random selection is sufficient at any message volume this system will reach. A test nonetheless accumulates every nonce issued under each key and asserts zero collisions. The check costs almost nothing, and nonce reuse would be unrecoverable.

4) Overhead

The wire overhead is a 16-byte tag plus a 16-byte nonce, independent of payload size:

[
|c|_{\mathrm{wire}} = |p| + 32\text{ bytes}.
\tag{29}
]

For a typical 96-byte sensor reading this is a 33% expansion, falling to 6.3% at 512 bytes. The relative cost is therefore worst exactly where agricultural telemetry sits, which is an argument for batching several readings per message and a measurement we report rather than assume.

## H. Runtime Pipeline and Feature Provenance

This subsection addresses gap G6, and what it reports is a limitation rather than a result.

A payload such as

{"deviceId":"soil01","temperature":24.8,"soilMoisture":42.5}

contains no flow-derived features at all. Feeding these fields to a model trained on CICIoT2023 and reporting a detection outcome would be meaningless. The fields are application data; the model consumes network measurements. They are different objects and the code keeps them apart.

Each simulated message therefore carries two planes. The application plane is the JSON payload, and it is what Ascon protects. The network plane is a flow-feature vector drawn from held-out CICIoT2023 records never seen during training, standing in for the network conditions under which that message was transmitted, and it is what the GRU classifies. The adapter records the origin of every feature the model consumes. No value is invented.

We considered two alternatives and rejected both. Synthesising network features with a generative model cannot be verified, and it would evaluate the detector against its own generator’s artefacts. Instrumenting a live MQTT broker recovers only part of the CICIoT2023 feature set without the original extractor, and mixing measured features with replayed ones introduces a distribution shift we could not document; we list it as future work.

The runtime demonstration establishes architectural correctness. It shows that the pipeline separates the two planes, routes on the verdict, encrypts and verifies correctly, and never places a malicious-verdict payload on the cloud path. It does not show that a CICIoT2023-trained model would detect attacks against a live agricultural MQTT deployment. That claim would need capture and feature re-extraction on the target network, and we do not make it.

## I. Evaluation Protocol

1) Baselines

Five, reported for every configuration: a random forest on single records (motivated by S13); a multilayer perceptron on single records, which isolates the contribution of recurrence; a centralised GRU, giving the upper bound attainable by pooling; three local-only GRUs, giving the lower bound attainable without federating; and the federated global GRU. The centralised and local-only pair bracket the federated result and together answer G4.

2) Metrics

For the eight-class task we report macro-averaged precision, recall, and

\frac{1}{C}
\sum_{c=1}^{C}
\frac{2P_cR_c}{P_c+R_c},
\tag{30}
]

together with per-class F1, balanced accuracy, the Matthews correlation coefficient, and the confusion matrix.

For the binary projection we report precision, recall, PR-AUC, and the false-positive rate

\frac{FP}{FP+TN}.
\tag{31}
]

FPR is treated as a first-class metric because in an operational detector a high false-alarm rate is the failure that causes the system to be switched off. We report accuracy alongside these, never on its own.

We also report federated convergence as macro-F1 against round, the client-to-global gap, communication bytes per round by Equation (22), Ascon encrypt and decrypt latency at the median and 95th percentile, ciphertext expansion, and per-stage runtime latency.

3) Ablations

Six, each isolating one variable: window length (W \in {1, 8, 16, 32}); heterogeneity (\alpha \in {0.1, 0.5, 100}); local epochs (E \in {1, 3, 5}); feature count (F \in {8, 12, 16, 24, F_0}); weighted against unweighted aggregation; and rounds (R) up to 20.

Table VI — Risks with a design consequence. L: likelihood, I: impact.

Risk

L

I

Control

R1 Memory exhaustion on load

H

M

Chunked reads, per-class caps, float32

R2 Rare classes collapse to zero recall

H

M

Eight-class taxonomy, weighted loss, per-class reporting

R3 Duplicate-row leakage inflates results

H

H

Dedup before split; blocking test

R4 FedAvg diverges under non-IID data

M

M

Fewer local epochs, layer norm, FedProx fallback; negative result reported

R5 Ascon variant mismatch

M

M

Known-answer tests as a gating check

R6 Nonce reuse

L

H

CSPRNG per message; collision assertion

R7 Near-perfect results accepted uncritically

M

M

Stated expectation, Section III-I; leakage audit

4) Statistical protocol

Every headline result is run over three seeds and reported as mean ± standard deviation. We do not report single-run numbers as findings. Seeds, library versions, hardware, and hyperparameters are captured in a per-run manifest.

5) A stated expectation

Published results on CICIoT2023 routinely exceed 98–99% accuracy. Near-ceiling accuracy on this benchmark is a known property of the data, produced by the imbalance of Equation (6) and by the separability of the flooding classes, and it is not evidence of a superior model.

We state in advance that results landing in that range will be treated as expected, and we will direct attention instead to macro-F1 on the rare families and to FPR, where the differences between methods actually show up. Committing to this before the experiments are run guards against post-hoc rationalisation and tells a reader which of our numbers are worth examining.

## J. Feasibility, Risk, and Ethics

1) Feasibility

Every component has a mature reference implementation, and the closest prior work S10 has already demonstrated that a GRU model trained under weighted federated averaging on CICIoT2023 is workable. The binding constraint is memory rather than computation: the full corpus does not fit, which the capped subsample of Section III-B resolves, bringing the working set to a few gigabytes.

By Equation (19) the model is small enough for CPU-only training, so no GPU is required and no cloud tenancy is purchased. The project runs on a single workstation with open-source software.

2) Risks

Table VI lists the risks that carry a design consequence rather than the full register. Two deserve comment.

R3 is treated as a blocking condition: the duplicate-leakage test gates the pipeline, because a result computed on a leaking split is worse than no result. R4 admits the possibility that federation underperforms the local-only baseline under strong heterogeneity. That would still be a legitimate finding, and it is the quantity an operator most needs to know; we will report it as such rather than tuning until it disappears.

3) Ethics

The work involves no human participants and no personally identifiable information; CICIoT2023 is machine-generated traffic from a controlled testbed, used under the terms its publishers set for academic research and cited accordingly. No live attack traffic is generated: every malicious instance is a replayed record, and nothing is directed at any real network or third party. The artefact is defensive, and no attack tooling or evasion technique is produced. Key material is excluded from version control, and demonstration keys are labelled as such.

Two commitments concern honesty of claims rather than conduct. We do not claim that federated learning by itself provides privacy; the property we claim is the narrower one that raw records are not transmitted. And we do not claim that application-layer fields are equivalent to flow-derived features, for the reasons given in Section III-H. We state both because an undocumented limit invites the kind of overconfident deployment this domain cannot afford.

4) Work plan

Development proceeds in seven gated phases, none beginning before its predecessor meets an exit criterion: characterisation report; leakage-controlled preprocessing and feature selection; centralised GRU with full evaluation; three clients with weighted FedAvg; telemetry simulation and adapter; Ascon and alerting paths; end-to-end integration.

Phase 3 is a hard gate: federation is not begun until single-client detection is proven, because debugging an aggregation fault and a modelling fault simultaneously is substantially harder than debugging either alone.

Responsibilities divide into four roles: data and features, model and federation, IoT and security, and integration and quality. Each member tests their own modules, and a weekly checkpoint reviews the gate criterion and the risk register.


## TABLE V

*REVIEWED CORPUS (S1–S20) ADMITTED UNDER THE PROTOCOL OF SECTION II-B. FL: FEDERATED LEARNING. AEAD: AUTHENTICATED ENCRYPTION [URL 🔗](#page-0)*

*WITH ASSOCIATED DATA. “RELEVANCE” STATES WHAT THE STUDY CONTRIBUTES TO THE PRESENT DESIGN OR TO THE GAP ANALYSIS. [URL 🔗](#page-0)*

| Focus | Method | Data / platform | Relevance to this work |
| --- | --- | --- | --- |
| S1 [1] IoT attack benchmark | Testbed capture, 105 devices CICIoT2023 |   | Supplies the dataset; establishes the imbalance and the |
|   |   |   | packet-window record semantics |
| S2 [2] Decentralised training | FedAvg | Image benchmarks | Aggregation rule adopted, Eq. (21) |
| S3 [3] Heterogeneous federation FedProx |   | Mixed | Explains client drift under non-IID data; documented fallback if |
|   |   |   | FedAvg diverges |
| S4 [4] FL for IoT | Survey | — | Frames bandwidth and governance motivation for federation |
| S5 [5] Federated IDS | Systematic review | — | Umbrella reference; source of limitations L1–L4 |
| S6 [6] IoT IDS | Federated deep learning | IoT flow data | Confirms feasibility of federated deep detectors at the edge |
| S7 [7] Industrial IoT IDS | Privacy-preserving FL | Industrial data | Quantifies the privacy–utility trade-off we place out of scope |
| S8 [8] IoT IDS | Optimised FL | IoT benchmarks | Reports GRU clients sharing parameters to a central aggregator |
| S9 [9] IoT IDS | FL + TabTransformer | N-BaIoT, UNSW-NB15, | Alternative architecture on the same benchmark |
|   |   | CICIoT2023 |   |
| S10 [10] IoT IDS | FL, CNN-GRU / LSTM-GRU, | CICIoT2023, | Closest prior work; validates GRU + wFedAvg on our dataset; |
|   | weighted FedAvg | FLNET2023 | stops at the label |
| S11 [11] IoT IDS | FedAvg vs. | IoT dataset | Evidence on aggregation-rule choice |
|   | FedAvg-momentum |   |   |
| S12 [12] Distributed IoT security | Adaptive weighting + | Tabular IoT, | Shows DP cost is modest; supports our future-work path |
|   | per-device DP | Edge-IIoTset |   |
| S13 [13] IoT IDS | Nine ML/DL models; three | UNSW-NB15 | Dissenting evidence: tree ensemble beat GRU/LSTM; motivates |
|   | aggregation rules |   | our RF baseline |
| S14 [14] Lightweight AEAD | Ascon v1.2 design | — | Primitive specification and design rationale |
| S15 [15] Ascon hardware | ASIC realisation | Constrained hardware | Evidence of suitability for field-node cost budgets |
| S16 [16] Ascon performance | Benchmarking | AI-enabled IoT devices | Latency and resource evidence on IoT-class hardware |
| S17 [17] Lightweight encryption | Ascon, PRESENT, SIMON, | MQTT and CoAP, | Directly comparable protocol-level measurement methodology |
| over IoT protocols | SPECK, ChaCha20 | ESP32 |   |
| S18 [18] Industry 4.0 security | Ascon AEAD | MQTT | Ascon compared against AES-GCM over MQTT |
| S19 [19] IoT/robotic data security Ascon-AEAD128 variant |   | MQTT | Confirms the standardised variant is usable over MQTT |
| S20 [20] Precision agriculture | Lightweight cipher for | Agricultural IoT | Establishes the agricultural threat scenario for Channel 2; no |
|   | irrigation |   | detection layer |

TRAINING PLANE

Fig. 2. System architecture. In the training plane, only the parameter vector θk and the sequence count nk cross the client boundary; raw records never do. In the runtime plane, the application payload and the network-level features are handled as separate objects, and the classifier verdict selects between two disjoint output paths. The malicious path holds no reference to the cloud transport.

We deliberately do not apply principal component analysis. Components are linear mixtures of features, which destroys the column-level interpretability an operator needs in order to

act on an alert, and would make the runtime adapter of Sec- tion III-H impossible to specify honestly. The interpretability cost is real once an operator has to act on the output. [URL 🔗](#page-0)


## D. Temporal Sequence Construction

This section addresses gap G5, and the resolution is a stated assumption rather than a technique.

The distributed CSV files are not expected to carry a

timestamp or a flow identifier. Without one, per-device chrono- logical ordering cannot be recovered, and a sliding window over rows is positional rather than temporal. Much of the surveyed work slides a window anyway, which claims more than the data supports.

We construct windows only over contiguous same-label runs within a single source file, preserving whatever ordering the capture-to-CSV extraction imposed, and we record this as an assumption instead of presenting it as recovered chronology. Rows are never shuffled before windowing; shuffling is applied to complete sequences when batches are formed. Three rules follow. Training windows do not cross a label boundary, so supervision stays unambiguous. The sequence label is that of the final record, yi = yi+W−1, which matches the online semantics of deciding what is happening now given recent history. A separate held-out set of mixed windows spanning a benign-to-attack boundary is reserved for evaluation only, and is used to measure detection latency: how many records into an attack the verdict takes to flip.

Given runs of lengths {Lr}, the number of sequences is

Nseq = X max 0, Lr −W+ 1

which makes explicit that short runs contribute nothing once W grows large. That cost of increasing the window is easy to overlook. We sweep W ∈ {1, 8, 16, 32}, and include W = 1 as an ablation. If it matches W = 16, recurrence has not earned its place in the architecture, and we will say so.

## E. Detection Mode

---

# IMPLEMENTATION DEVIATION FROM THIS DESIGN (post-review, 2026-09-20)

This section is an **addendum**, added after the review-2 checkpoint, recording a deliberate
divergence between the implementation and the design specified in Sections I–III above. The
original text is left intact as the historical record of what was designed; this section states
what the code now does instead, and why. Per the project's operating rule (`CLAUDE.md` Golden
Rule 1) the divergence is flagged here rather than silently reconciled into the body of the
paper. Where the two disagree, **this section describes the code; Sections I–III describe the
original design.**

## What changed

The paper couples the detector's verdict to Ascon-AEAD128 protection of **Channel 2**, the
gateway-to-cloud telemetry channel (Section I-B; Eq. 5's `Enc_ke` on the benign branch; the
authenticated-encryption design of Section III-G; the associated-data tuple of Eq. 27; the
overhead of Eq. 29). Two things change:

1. **Ascon is removed from Channel 2.** The benign-verdict telemetry payload is now sent to the
   cloud receiver **in the clear**. Confidentiality and integrity of the cloud payload are
   assumed to be provided by mechanisms **outside the scope of this codebase** (for example a
   TLS-terminated transport, or an application-layer envelope the deployment already runs). The
   system therefore no longer implements Eq. 5's `Enc_ke`, nor the wire overhead of Eq. 29, on
   the telemetry channel. **What is retained from that design is the verdict-selects-path
   structure itself** (gap G1): the benign and malicious paths remain disjoint *by construction*,
   the malicious-path handler still holds no reference to the cloud transport, and the test that
   asserts no malicious-verdict payload can reach the cloud (`tests/test_path_disjointness.py`)
   remains a gating invariant. Replay detection via the monotonic counter (Section III-G2) is
   also retained on the cloud receiver, since it does not depend on encryption.

2. **Ascon is applied to Channel 3, the node-to-aggregator model-update channel** (Section I-B,
   "Channel 3"), which the paper discusses but explicitly does **not** solve ("federated learning
   addresses [reconstruction] only partially and [poisoning] not at all"). Every model-weight
   blob crossing the client↔aggregator boundary — in **both** directions of **every** round
   (server→client broadcast and client→server upload) — is now encrypted with Ascon-AEAD128 on
   send and decrypted on receive. Decryption returns ⊥ (Eq. 26) on any tampered blob, surfaced as
   a hard failure that aborts the round rather than aggregating unverified parameters. The
   primitive, its KAT-conformance gate (Section III-G1, R5), and its nonce discipline (Section
   III-G3, R6) are unchanged; only the channel they protect has moved.

## Associated data and keys for Channel 3

The associated-data tuple mirrors Eq. 27's shape but is specialised to this channel:

    a = ⟨client_id, round_index, direction, schema_version⟩

`direction ∈ {upload, broadcast}` binds each ciphertext to its leg of the round, so a blob
produced for one leg cannot be replayed as the other; `round_index` gives the ordering/replay
protection that Eq. 27's monotonic counter gives on the telemetry channel. The wire encoding is
the same length-prefixed (TLV) framing already resolved for Eq. 27 (`docs/plans/phase6-ad-serialization.md`),
shared in code so both AD types use one tested implementation.

Keys follow an **extension of assumption A5** ("Ascon keys are pre-shared out of band"): one
pre-shared 128-bit symmetric key per client↔server pair, reused for both directions across all
rounds with a fresh CSPRNG nonce per message. Production key management remains out of scope
(Section III-J3); the driver scripts generate demo keys inline and never write key material to a
manifest.

## Effect on the communication-cost model (Eq. 22)

Eq. 22, `B_round = 2K|θ|b`, is unchanged as the model of the raw parameter traffic. The
deviation adds a **constant per-round overhead**: a 16-byte nonce plus a 16-byte tag on each of
the two legs per client, i.e. `2K·32` bytes per round (192 bytes for K=3), negligible against the
~0.77 MiB/round the parameters themselves cost. Detection-quality results (macro-F1, the G4
bracket, per-class F1, FPR) are **unaffected**: the encryption round-trips losslessly, so the
tensors a client trains on are bit-identical to those it would receive without it.

## What this deviation does and does not claim

It adds confidentiality and tamper-evidence to the model-update *wire*. It does **not** provide
secure aggregation, does **not** defend against a curious-but-honest aggregator inspecting the
plaintext it legitimately decrypts (assumption A4 stands), and does **not** defend against
gradient-inversion or a poisoned update — all of which Section I-F and Section III place out of
scope, and all of which remain out of scope. The narrower privacy claim of the paper ("raw
records are not transmitted") is likewise unchanged.
