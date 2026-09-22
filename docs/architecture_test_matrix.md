# Architecture test matrix — what each component must guarantee, and the test that proves it

This is the walkthrough document for software testing of the Phase 8 architecture. It maps every
block of the system to the guarantee it owes, the attack that would break that guarantee, and the
test that demonstrates the attack failing.

**How to read a row.** Each row is written from the attacker's side. The test passes when the
attacker *fails* — a rejected frame, a dropped message, a `None` where a decrypted model would
have been. A green suite therefore means "every listed attack was attempted and denied", not
merely "the code ran".

Total suite: **437 tests**. The adversarial walkthrough is
[`tests/test_architecture_adversarial.py`](../tests/test_architecture_adversarial.py) (12 tests,
one block per component); the deeper property suites it cites are listed per component below.

## The architecture under test

```
  ESP32 ×3          MQTT              Raspberry Pi                 Laptop
  soil01..03  ──A──▶ broker ──B──▶  ┌──────────────────┐         ┌───────────────┐
  (farm 1)          farm1           │ provenance (C)   │         │  master GRU   │
                                    │ local GRU        │◀──D────▶│  (aggregator) │
  ESP32 ×3          MQTT            │ verdict          │ Ascon   │   FedAvg (E)  │
  soil04..06  ──A──▶ broker ──B──▶  │                  │ sealed  └───────────────┘
  (farm 2)          farm2           └───────┬──────────┘          both directions
                                            │
                              benign ──F──▶ TLS ──▶ cloud receiver
                              malicious ──▶ local alert sink (never the cloud)
```

Blocks **A–F** below follow a reading's path. Block **G** is the cryptographic core that **D**
and **F** rest on; block **H** covers the learning-pipeline invariants that make a verdict
meaningful in the first place.

---

## A. Sensor contract — ESP32 → farm broker

The node publishes `farm/<farm>/sensor/<device>` as JSON carrying a monotonic `seq`. Anything
that does not satisfy the contract is dropped rather than guessed at.

| Attack | Expected denial | Test |
| --- | --- | --- |
| Replay a captured reading verbatim to mask a live one | Dropped: `seq` is monotonic per device | `test_attacker_replaying_a_captured_sensor_reading_cannot_advance_the_stream` |
| Forge a reading with no `seq`, or a non-JSON body | Dropped, not inferred | `test_attacker_forging_a_reading_the_node_never_sent_is_dropped_when_malformed` |
| Drive a device into a scenario it has no records for | Refused rather than synthesised | `test_scenario_without_records_refuses_rather_than_synthesising` (`test_mqtt_source.py`) |

Supporting properties: `test_readings_are_parsed_ordered_and_replay_filtered`,
`test_malformed_foreign_and_mismatched_messages_are_dropped` (`test_mqtt_source.py`).

## B. Farm isolation — broker → Pi

Each Pi consumes its own farm only. In the Compose topology the farms also have separate brokers,
so this is defence in depth.

| Attack | Expected denial | Test |
| --- | --- | --- |
| Publish into another farm's topic from a compromised broker | Ignored; the Pi's own farm still works | `test_attacker_publishing_into_another_farms_topic_is_ignored` |
| Point farm 1's Pi at farm 2's broker | Structurally prevented: each Pi is wired to its own broker | `test_compose_topology_matches_the_plan` (`test_software_twin.py`) |

## C. Feature provenance boundary (G6) — on the Pi

Network features come **only** from held-out CICIoT2023 records — never from the JSON payload,
never synthesised. This is the boundary that keeps the demo honest.

| Attack | Expected denial | Test |
| --- | --- | --- |
| Stuff network-feature values (`Header_Length`, `IAT`, …) into the sensor JSON | Impossible by signature: the adapter takes a stream index and a scenario, and has no parameter a payload could reach | `test_attacker_cannot_inject_network_features_through_the_sensor_payload` |
| Request an `attack` scenario on a node with no held-out attack pool | `ValueError` — refuses rather than fabricating traffic | `test_attacker_cannot_make_the_adapter_synthesise_a_scenario_it_has_no_records_for` |
| Draw features from records that were also used in training | Scenario pools draw only from held-out records | `test_scenario_pools_draw_only_held_out_records_of_that_kind` (`test_mqtt_source.py`) |

## D. Sealed weight channel — Pi ⇄ aggregator **(the headline claim)**

Every client update and every global-state download is an Ascon-AEAD128 frame under a per-client,
per-direction key, with associated data `⟨client_id, round, direction, schema_version⟩`. The
aggregator never applies a frame that opens to `⊥`.

| Attack | Expected denial | Test |
| --- | --- | --- |
| **Capture a frame and try to decrypt it without the key** | Every wrong key yields `⊥`; the honest key still opens it | `test_eavesdropper_without_the_key_cannot_read_the_weights` |
| **Skip decryption — scan the captured bytes for the weights** | No tensor appears in the clear anywhere in the frame; only the AD is readable, and that is authenticated, not secret | `test_a_captured_frame_does_not_contain_the_weights_in_the_clear` |
| Compare two frames to learn whether the model changed | Fresh CSPRNG nonce per message: identical weights seal to different ciphertext of the same length | `test_identical_weights_sealed_twice_do_not_produce_the_same_frame` |
| Downgrade the channel by sending unsealed weights | Rejected before any state is deserialized — there is no plaintext path | `test_attacker_cannot_downgrade_the_channel_by_sending_unsealed_weights` |
| Re-address pi-1's update as pi-2's to steer the average | Rejected: keys are per client, and the AD names the client | `test_attacker_cannot_reuse_a_client_frame_against_a_different_client` |
| Flip any single bit of nonce, AD, ciphertext or tag | `⊥`; nothing applied | `test_single_bit_tamper_anywhere_yields_bottom` (`test_weight_channel.py`) |
| Replay a round-*r* frame at round *r+1* | Rejected | `test_round_r_frame_is_rejected_at_round_r_plus_1` |
| Reflect an uplink frame back as a downlink | Rejected: direction is bound into the AD | `test_uplink_frame_reflected_as_downlink_is_rejected` |
| Reuse a nonce under one key | Structurally prevented by a shared registry | `test_sealers_share_one_registry_and_never_reuse_a_nonce` |
| Use the same key in both directions | Refused at construction | `test_channel_keys_must_differ_per_direction` |
| Present a telemetry-shaped AD on the weight channel | Mutually unparseable | `test_weight_ad_and_telemetry_ad_are_mutually_unparseable` |

Correctness alongside security: `test_fedavg_over_channel_equals_in_process_fedavg` and
`test_loopback_federation_matches_in_process_bit_for_bit` prove the sealing changes *nothing*
about the result — networked FedAvg equals in-process FedAvg bit-for-bit.

## E. Aggregator integrity — the master GRU

| Attack | Expected denial | Test |
| --- | --- | --- |
| Replay a client's own frame inside the same round to double-count it | Rejected by the transport: the opener accepts only strictly newer rounds. Defence in depth — the server also keys updates by client id | `test_replaying_a_clients_own_frame_within_a_round_cannot_inflate_its_weight` |
| Edit an honest client's `n_k` in flight to skew Eq. (21) | `n_k` is authenticated; any edit yields `⊥` and the update is dropped whole | `test_attacker_cannot_tamper_with_an_honest_clients_declared_n_k` |
| Get a tampered update applied to the global model | Dropped, not applied; aggregation proceeds without it | `test_tampered_update_frame_is_dropped_not_applied` (`test_sealed_server.py`) |
| Send a frame from an unknown client id | Rejected and logged | `test_frame_for_another_client_is_rejected` |

## F. Cloud path (G1) — benign over TLS, malicious never

Path disjointness is **structural, not conventional**: the malicious-path handler holds no
reference to the cloud transport, so a malicious verdict cannot reach the cloud even by mistake.

| Attack | Expected denial | Test |
| --- | --- | --- |
| Get a malicious verdict onto the cloud transport | Impossible: the alert sink's constructor takes no transport, and holds no reference to one | `test_alert_sink_constructor_takes_no_transport`, `test_alert_sink_holds_no_reference_to_any_cloud_transport` |
| Run a malicious-only stream and watch the wire | Zero payloads emitted to the cloud | `test_malicious_only_run_emits_zero_encrypted_payloads`, `test_malicious_only_run_emits_nothing_to_the_cloud` |
| Replay a benign envelope to the cloud receiver | Rejected; receiver is replay-guarded per device | `test_receiver_rejects_replays_unknown_edges_and_garbage` |
| Post as an edge the receiver does not know | Rejected | same test |

## G. Cryptographic core — underpins D and F

| Guarantee | Test |
| --- | --- |
| The Ascon-AEAD128 backend matches the NIST SP 800-232 known-answer vectors (gate R5) | `test_backend_matches_sp800_232_vectors` (`test_ascon_kat.py`) |
| Any modified ciphertext or AD decrypts to `⊥` | `test_single_bit_ciphertext_mutation_rejected`, `test_single_bit_associated_data_mutation_rejected` |
| Zero nonce reuse per key; deliberate reuse raises | `test_no_nonce_reuse_under_one_key`, `test_deliberate_reuse_raises` |

No primitive is hand-rolled (III-G1): the backend is a maintained library gated behind the KAT.

## H. Learning-pipeline invariants — is the verdict meaningful?

Not adversarial, but part of any honest software test: a detector that leaks its test set or
mis-weights its clients produces numbers that mean nothing.

| Invariant | Test |
| --- | --- |
| Dedup **before** split; no record hash in both partitions (leakage gate R3) | `test_no_record_hash_in_both_partitions` |
| FedAvg weight `n_k` is training **sequences**, not raw rows (Eq. 21) | `test_weight_uses_sequence_count_not_row_count` |
| Federated scaler equals the pooled scaler without pooling data (Eqs. 23–24) | `test_combined_stats_equal_pooled_fit` |
| Default model reproduces Eq. (19) = 33,800 parameters | `test_default_sizing_is_exactly_33800_parameters` |

---

## Running the tests

```bash
# everything (437 tests)
./.venv/bin/python -m pytest

# the adversarial walkthrough alone -- the one to project on a screen
./.venv/bin/python -m pytest tests/test_architecture_adversarial.py -v

# one component at a time, e.g. the sealed weight channel
./.venv/bin/python -m pytest tests/test_weight_channel.py tests/test_sealed_server.py -v
```

`tests/test_software_twin.py::test_compose_file_is_valid_for_docker` is skipped unless `docker`
is on `PATH`; everything else runs without Docker.

## Demonstrating it live on the twin

The Compose topology asserts the same properties at runtime, and each container prints the
evidence. From the loopback run of 2026-09-22 (the Docker run prints the same lines, tagged
`platform=docker-arm64-simulation`):

- **Aggregator** — `[round 1/2] ... | up 407,752 B | down 407,758 B | rejected so far 0`.
  Every weight frame in both directions was sealed and every one authenticated.
- **Pi runtime** — `G1: malicious readings reaching the cloud = 0 by construction; cloud accepted
  111 == benign verdicts 111: True`, alongside `benign 111 -> cloud ... malicious 84 -> alerts 84`.
- **Cloud receiver** — `accepted 111 | rejected 0`, replay-guarded per device.

To show the channel refusing an attack live rather than in a unit test, tamper with a frame in
flight — `test_tampered_frame_on_the_wire_is_dropped_and_client_recovers`
(`test_node_loopback.py`) does exactly this over a real socket and shows the client recovering on
the next round.

## What is deliberately **not** defended

Stating this plainly is part of the test report. These are out of scope by project decision
(see `CLAUDE.md`), not oversights:

- **A dishonest client.** `n_k` is authenticated against outside tampering, but a client holding
  a valid key can declare an inflated `n_k` under it. Defending that is Byzantine-robust
  aggregation — out of scope. The architecture's trust boundary is the **network**, not the client.
- **Inference attacks on the weights themselves** — gradient inversion, membership inference.
  Differential privacy and secure aggregation are out of scope; Ascon protects the weights in
  transit, not against what a legitimate aggregator can infer from them.
- **Production key management.** Demo keys are generated per run, labelled as demo, and
  gitignored. There is no attestation, rotation or HSM story.
- **The cloud TLS certificate** is self-signed for the demo.
- **G6 and the runtime demo.** The runtime path shows architectural correctness only: features
  are replayed from held-out records, so detection quality on the demo stream is not a field
  measurement. See the docstring in `telemetry/provenance.py`.
