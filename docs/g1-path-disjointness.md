# G1 — path disjointness: why it matters, how it is enforced, and how it was tested

G1 is the project's central safety guarantee: **a reading the detector judges malicious must
never reach the cloud.** This document states the threat it addresses, why it is enforced
*structurally* rather than by convention, every layer at which it is tested, and the measured
results — including one run where G1 "passed" and the pass meant nothing.

---

## 1. What G1 claims

> The malicious-path handler holds **no reference** to the cloud transport — structurally, not
> by convention. Zero encrypted payloads may be emitted on a malicious-only run.

Two halves, and both are necessary:

- **Structural half** — the object that handles malicious verdicts *cannot* reach the cloud
  channel. Not "does not", *cannot*.
- **Behavioural half** — on a run containing only malicious verdicts, the cloud receives nothing
  at all.

---

## 2. Why it matters

The architecture sends benign agricultural telemetry to a cloud receiver over TLS, and the local
GRU on each Raspberry Pi decides what is benign. If a malicious reading slipped onto that path,
three things would go wrong at once:

1. **The detector's entire purpose is defeated.** The system exists to stop bad data at the farm
   edge. A detector that forwards what it just flagged is not a detector; it is a relay.
2. **The cloud becomes a propagation vector.** Attack traffic exported to an external service is
   no longer contained to one farm. It is now in a system shared across farms.
3. **The security claim becomes unverifiable.** "Only benign readings leave the farm" is either
   true for every reading or it is not a property at all. One leak makes the claim false, and a
   claim that holds *usually* cannot be stated in a paper.

This is also why the malicious path is **local only**: a malicious verdict raises an alert on the
Pi and goes nowhere else. There is no "quarantine upload", no "send it for analysis". Those would
all be cloud paths wearing a different name.

---

## 3. How it is enforced — structurally

The naive approach is a rule: *"remember not to send malicious payloads to the cloud."* That is a
comment, a code-review convention, and a bug waiting to happen — one refactor, one well-meaning
"let's log malicious readings centrally" feature, and the guarantee is silently gone.

Instead the guarantee is built into the object graph. Only **one** object ever holds the cloud
transport:

```
                  ┌──────────────────────────────────────┐
                  │           VerdictRouter              │
                  │   self._cloud  ◄── the ONLY holder   │
                  │   self._alert                        │
                  │   self._cipher                       │
                  └───────────┬──────────────┬───────────┘
                              │              │
                  verdict_benign=True   verdict_benign=False
                              │              │
                              ▼              ▼
              ┌───────────────────────┐  ┌──────────────────────┐
              │   CloudTransport      │  │     AlertSink        │
              │   encrypt + send      │  │  __init__(self)      │
              │                       │  │  NO dependencies     │
              │                       │  │  NO transport        │
              └───────────────────────┘  └──────────────────────┘
```

`routing/router.py` carries the comment that makes the intent explicit at the one place it
matters:

```python
# The cloud transport lives ONLY here; it is never handed to `alert`.
self._cloud = cloud
self._alert = alert
```

And `routing/alert_sink.py` takes **no constructor parameters at all**:

```python
class AlertSink:
    def __init__(self) -> None:
        self.alert_count = 0
        self._alerts: list[dict[str, str]] = []
```

Its module docstring states the rule for whoever edits it next:

> Do NOT add a CloudTransport (or anything that can encrypt/transmit toward the cloud) to this
> class — doing so would silently break the core guarantee of the architecture.

**The consequence:** there is no code path from a malicious verdict to the cloud, because the
object handling malicious verdicts has nothing to reach it with. Breaking G1 requires changing
`AlertSink`'s constructor signature — which three tests immediately fail on.

---

## 4. How it is tested — four layers

### Layer 1: structural, by introspection — `tests/test_path_disjointness.py`

**`test_alert_sink_constructor_takes_no_transport`** inspects the signature itself:

```python
params = [p for p in inspect.signature(AlertSink.__init__).parameters if p != "self"]
assert params == []
```

This fails the moment anyone adds *any* dependency, before the question of whether it is a cloud
transport even arises. It is deliberately stricter than necessary — the cheapest way to keep a
guarantee is to make the shape of the violation impossible.

**`test_alert_sink_holds_no_cloud_reference`** then checks the live object's attributes:

```python
for name, value in vars(alert).items():
    assert not isinstance(value, MockCloudReceiver)
    assert not isinstance(value, CloudTransport)
```

This catches an assignment made after construction, which the signature test cannot see.

### Layer 2: behavioural — the malicious-only run

**`test_malicious_only_run_emits_zero_encrypted_payloads`** routes five malicious verdicts
through the **real** `VerdictRouter` with a real `AsconAEAD128` cipher, then asserts:

```python
assert cloud.received_count == 0
assert cloud.rejected_count == 0  # nothing was even SENT to the cloud to reject
assert alert.alert_count == 5
```

The second assertion is the subtle one. `received_count == 0` alone would also be satisfied by a
system that sends malicious payloads and has the receiver reject them — which is **not** G1. G1
requires that nothing is transmitted at all. `rejected_count == 0` is what distinguishes
"structurally unreachable" from "sent and refused".

These tests carry `@pytest.mark.gating`, so they are part of the phase gate rather than ordinary
coverage.

### Layer 3: the real transport — `tests/test_tls_path.py`

Layers 1–2 use an in-process mock receiver. These six tests repeat the claim against the actual
TLS path, including `test_malicious_only_run_emits_nothing_to_the_cloud`,
`test_alert_sink_holds_no_reference_to_any_cloud_transport`, replay/unknown-edge/garbage
rejection, and a real HTTP round trip on loopback. The router in `routing/tls_path.py` is the only
holder of the cloud transport there too.

### Layer 4: end to end, across process and network boundaries

`tests/test_software_twin.py::test_headless_twin_end_to_end` runs the whole pipeline headlessly.
The containerised twin then runs it across real processes on a real network, with the cloud
receiver keeping **its own independent count** — so the Pi is not the only witness to its own
behaviour.

**64 of 64** hardware-path tests pass in total: weight channel, node loopback, software twin,
TLS path, path disjointness, Ascon KAT, Ascon tamper, nonce collision, MQTT source, and the
adversarial architecture suite. Full suite: **438 passed, 0 skipped**.

---

## 5. Results

### The runtime verifications, in order

| Run | Malicious readings | Malicious at cloud | Benign delivered | Verdict |
| --- | --- | --- | --- | --- |
| Phase 7 end-to-end | 55 | **0** | 0 | ✅ |
| 2026-09-19 loopback | 119 | **0** | 76 of 76 | ✅ |
| Twin, first attempt | **1** | 0 | 34 | ⚠️ **vacuous — see §6** |
| Twin, second run | 491 | **0** | 793 of 793 | ✅ |
| **Twin, R = 20 run** | **4,239** | **0** | **8,924 of 8,924** | ✅ |

### The R = 20 run in detail

| | pi-1 | pi-2 | total |
| --- | --- | --- | --- |
| Readings received | 6,626 | 6,627 | 13,253 |
| Still buffering (< 16 for that device) | 45 | 45 | — |
| Benign → cloud | 5,421 sent, **0 failed** | 3,503 sent, **0 failed** | **8,924** |
| Malicious → local alert | **1,160 → 1,160** | **3,079 → 3,079** | **4,239** |
| **Malicious reaching the cloud** | **0** | **0** | **0** |

Two independent properties hold here, and they are worth separating:

- **The security property (G1):** 4,239 malicious readings, **0** on the cloud path.
- **The delivery property:** every one of the 8,924 benign readings arrived — and the cloud
  receiver's **own** count, written by a different process, reads `accepted 8924, rejected 0`,
  which is exactly 5,421 + 3,503.

Every malicious reading also produced exactly one alert — 1,160 → 1,160 and 3,079 → 3,079 — so
nothing was dropped on the way to the alert sink either.

---

## 6. The run where G1 passed and the pass was worthless

This is the most useful methodological finding in the G1 work, and it is recorded so it is not
repeated.

The first containerised run reported:

```
[pi-1] received 80 | benign 34 -> cloud (sent 34, failed 0) | malicious 1 -> alerts 1
[pi-1] G1: malicious readings reaching the cloud = 0; cloud accepted 34 == benign verdicts 34: True
```

**G1 "True" — on a single malicious reading.**

The cause: the virtual sensors begin publishing as soon as the broker is up, while the Pis are
still federating (~160 s), and **MQTT QoS 0 discards anything published before a subscription
exists.** Both Pis missed almost the entire attack window. The assertion was satisfied, the
property was barely exercised, and nothing meaningful was learned.

**"Zero malicious reached the cloud" is not evidence when only one malicious reading existed.**
A guarantee about a class of events is only tested as strongly as the number of events that
actually occurred.

The fix was to move the attack window inside the Pis' **post-federation** listening period
(`ATTACK_AFTER` after the federation finishes, not after wall-clock start). That took the count
from 1 → 491 → 4,239.

### And a second misread, in the opposite direction

The next attempt raised `MESSAGES` to 800 without raising `CLOUD_SECONDS`. The log then showed
190 benign sends "failing", which looked like a transport fault. It was not: the receiver had
reached its 600 s budget and exited cleanly while the Pis kept sending. Sent (355 + 288) still
equalled accepted (643) exactly.

**The distinction to carry into any future log reading:** a *security* property (no malicious
reading reaches the cloud) and a *delivery* property (every benign reading arrives) fail for
completely different reasons, and **only the first is a G1 claim.** Reading a delivery failure as
a G1 failure, or a vacuous G1 pass as a strong one, are both easy mistakes.

---

## 7. What G1 does and does not cover

**Covered:** no reading classified malicious is transmitted to the cloud, by any route, because
the handler for malicious verdicts cannot reach a transport. This holds regardless of verdict
*quality* — G1 is about routing, not about detection accuracy.

**Not covered, and deliberately so:**

- **Whether the verdict is correct.** G1 guarantees a malicious-*verdict* reading never leaves.
  If the model misclassifies an attack as benign, that reading goes to the cloud and G1 is still
  satisfied. Detection quality is a separate, measured question (macro-F1, FPR) and the FPR is
  reported first in every evaluation for exactly this reason.
- **Confidentiality of the benign path.** That is TLS's job on the Pi → cloud hop, and
  Ascon-AEAD128's job on the weight channel.
- **A live agricultural deployment.** The runtime demonstration classifies held-out CICIoT2023
  records paired to each message (the G6 boundary), so it shows architectural correctness, not
  detection on live farm traffic.

---

## 8. Where to look

| Thing | File |
| --- | --- |
| The malicious-path handler | `src/ascon_smart_agri/routing/alert_sink.py` |
| The one holder of the cloud transport | `src/ascon_smart_agri/routing/router.py` |
| The TLS benign path | `src/ascon_smart_agri/routing/tls_path.py` |
| Structural + behavioural tests | `tests/test_path_disjointness.py` |
| TLS-path tests | `tests/test_tls_path.py` |
| End-to-end twin test | `tests/test_software_twin.py` |
| Measured runtime results | `artifacts/manifest_phase8_runtime_pi-*.json`, `artifacts/manifest_phase8_cloud_receiver.json` |
| Twin walkthrough | `docs/software-twin-walkthrough.md` |
