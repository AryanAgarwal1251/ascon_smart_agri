"""Unit tests for the mock cloud receiver (Section III-A, III-G, A6).

IMPLEMENTATION DEVIATION FROM THE DESIGN PAPER (flagged per CLAUDE.md golden rule 1): this
receiver no longer decrypts anything -- see ``routing/cloud_sink.py``'s module docstring. Covers
what remains encryption-independent: malformed-metadata rejection, the edge-identity check, and
the load-bearing replay-window ordering (rejection must never advance replay state).
"""

from __future__ import annotations

from ascon_smart_agri.crypto.ascon_aead import AssociatedData
from ascon_smart_agri.routing.cloud_sink import MockCloudReceiver

AD = AssociatedData("edge01", "dev01", 0, "v1")


def test_a_well_formed_payload_is_accepted() -> None:
    receiver = MockCloudReceiver()

    receiver.send_plaintext("edge01", AD.to_bytes(), b"benign reading")

    assert receiver.received_count == 1
    assert receiver.rejected_count == 0
    assert receiver.accepted_payloads == [b"benign reading"]


def test_malformed_metadata_is_rejected_not_crashed() -> None:
    receiver = MockCloudReceiver()

    receiver.send_plaintext("edge01", b"\xff\x00not-valid-ad", b"payload")

    assert receiver.received_count == 0
    assert receiver.rejected_count == 1


def test_edge_id_param_must_match_the_metadatas_edge_id() -> None:
    """A message whose metadata says one edge_id but arrives labelled under another is rejected."""
    receiver = MockCloudReceiver()

    receiver.send_plaintext("edge02", AD.to_bytes(), b"payload")  # AD says edge_id="edge01"

    assert receiver.received_count == 0
    assert receiver.rejected_count == 1


def test_a_replayed_message_is_rejected_by_the_second_delivery() -> None:
    receiver = MockCloudReceiver()

    receiver.send_plaintext("edge01", AD.to_bytes(), b"payload")  # first: accepted
    receiver.send_plaintext("edge01", AD.to_bytes(), b"payload")  # replay: rejected

    assert receiver.received_count == 1
    assert receiver.rejected_count == 1


def test_a_rejected_message_never_advances_replay_state() -> None:
    """The load-bearing ordering: a malformed message claiming counter=5 must not consume that
    counter, so the LEGITIMATE message at counter=5 is still accepted afterwards."""
    receiver = MockCloudReceiver()
    ad5 = AssociatedData("edge01", "dev01", 5, "v1")

    receiver.send_plaintext("edge02", ad5.to_bytes(), b"payload")  # rejected: edge_id mismatch
    receiver.send_plaintext("edge01", ad5.to_bytes(), b"payload")  # the real message

    assert receiver.rejected_count == 1
    assert receiver.received_count == 1  # counter=5 was still available
    assert receiver.accepted_payloads == [b"payload"]


def test_two_devices_under_the_same_edge_do_not_block_each_other() -> None:
    receiver = MockCloudReceiver()
    ad_a = AssociatedData("edge01", "devA", 0, "v1")
    ad_b = AssociatedData("edge01", "devB", 0, "v1")

    receiver.send_plaintext("edge01", ad_a.to_bytes(), b"a")
    receiver.send_plaintext("edge01", ad_b.to_bytes(), b"b")

    assert receiver.received_count == 2
