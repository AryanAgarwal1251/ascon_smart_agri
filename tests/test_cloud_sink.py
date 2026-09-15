"""Unit tests for the mock cloud receiver (Section III-A, III-G, A6).

Covers Eq. (26)'s decryption contract end-to-end (accept only on a verifying tag), the demo
key-store boundary, and the load-bearing ordering: replay state must never move on a message
that fails verification.
"""

from __future__ import annotations

from ascon_smart_agri.crypto.ascon_aead import AsconAEAD128, AssociatedData
from ascon_smart_agri.routing.cloud_sink import MockCloudReceiver

KEY = b"\x11" * 16
AD = AssociatedData("edge01", "dev01", 0, "v1")


def _encrypt(payload: bytes, ad: AssociatedData = AD, key: bytes = KEY) -> tuple[bytes, bytes]:
    cipher = AsconAEAD128(key)
    nonce = b"\x22" * 16
    return nonce, cipher.encrypt(nonce, ad.to_bytes(), payload)


def test_a_correctly_encrypted_payload_is_accepted() -> None:
    receiver = MockCloudReceiver({"edge01": KEY})
    nonce, ciphertext = _encrypt(b"benign reading")

    receiver.send_encrypted("edge01", nonce, AD.to_bytes(), ciphertext)

    assert receiver.received_count == 1
    assert receiver.rejected_count == 0
    assert receiver.accepted_payloads == [b"benign reading"]


def test_an_unknown_edge_id_is_rejected_without_attempting_decryption() -> None:
    receiver = MockCloudReceiver({"edge01": KEY})
    nonce, ciphertext = _encrypt(b"payload")

    receiver.send_encrypted("edge99", nonce, AD.to_bytes(), ciphertext)

    assert receiver.received_count == 0
    assert receiver.rejected_count == 1


def test_a_tampered_ciphertext_is_rejected() -> None:
    receiver = MockCloudReceiver({"edge01": KEY})
    nonce, ciphertext = _encrypt(b"payload")
    tampered = bytes([ciphertext[0] ^ 0x01]) + ciphertext[1:]

    receiver.send_encrypted("edge01", nonce, AD.to_bytes(), tampered)

    assert receiver.received_count == 0
    assert receiver.rejected_count == 1
    assert receiver.accepted_payloads == []


def test_tampered_associated_data_is_rejected() -> None:
    receiver = MockCloudReceiver({"edge01": KEY})
    nonce, ciphertext = _encrypt(b"payload")
    wrong_ad = AssociatedData("edge01", "dev01", 999, "v1")  # different counter

    receiver.send_encrypted("edge01", nonce, wrong_ad.to_bytes(), ciphertext)

    assert receiver.received_count == 0
    assert receiver.rejected_count == 1


def test_edge_id_param_must_match_the_authenticated_ad() -> None:
    """Even a genuinely verifying (nonce, AD, ciphertext) is rejected if the caller's edge_id
    parameter disagrees with the authenticated AD's own edge_id."""
    receiver = MockCloudReceiver({"edge01": KEY, "edge02": KEY})
    nonce, ciphertext = _encrypt(b"payload")  # AD says edge_id="edge01"

    receiver.send_encrypted("edge02", nonce, AD.to_bytes(), ciphertext)

    assert receiver.received_count == 0
    assert receiver.rejected_count == 1


def test_a_replayed_message_is_rejected_by_the_second_delivery() -> None:
    receiver = MockCloudReceiver({"edge01": KEY})
    nonce, ciphertext = _encrypt(b"payload")

    receiver.send_encrypted("edge01", nonce, AD.to_bytes(), ciphertext)  # first: accepted
    receiver.send_encrypted("edge01", nonce, AD.to_bytes(), ciphertext)  # replay: rejected

    assert receiver.received_count == 1
    assert receiver.rejected_count == 1


def test_a_failed_verification_never_advances_replay_state() -> None:
    """The load-bearing ordering: a tampered message at counter=5 must not consume that counter,
    so the LEGITIMATE message at counter=5 is still accepted afterwards."""
    receiver = MockCloudReceiver({"edge01": KEY})
    ad5 = AssociatedData("edge01", "dev01", 5, "v1")
    nonce, ciphertext = _encrypt(b"payload", ad=ad5)
    forged = ciphertext[:-1] + bytes([ciphertext[-1] ^ 0x01])  # corrupt the tag

    receiver.send_encrypted("edge01", nonce, ad5.to_bytes(), forged)  # rejected: bad tag
    receiver.send_encrypted("edge01", nonce, ad5.to_bytes(), ciphertext)  # the real message

    assert receiver.rejected_count == 1
    assert receiver.received_count == 1  # counter=5 was still available
    assert receiver.accepted_payloads == [b"payload"]


def test_two_devices_under_the_same_edge_do_not_block_each_other() -> None:
    receiver = MockCloudReceiver({"edge01": KEY})
    ad_a = AssociatedData("edge01", "devA", 0, "v1")
    ad_b = AssociatedData("edge01", "devB", 0, "v1")
    nonce_a, ct_a = _encrypt(b"a", ad=ad_a)
    nonce_b, ct_b = _encrypt(b"b", ad=ad_b)

    receiver.send_encrypted("edge01", nonce_a, ad_a.to_bytes(), ct_a)
    receiver.send_encrypted("edge01", nonce_b, ad_b.to_bytes(), ct_b)

    assert receiver.received_count == 2
