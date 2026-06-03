"""Negative controls for SFT theory promotion.

These tests ensure the verification suite can reject attractive but invalid
interpretations instead of only confirming happy-path claims.
"""
import sft


def test_negative_control_rejects_sorted_spectrum_shortcut():
    result = sft.verify.verify_core_negative_controls()
    assert result.claim_id == "CORE-NEG-005"
    assert result.status == "PASS"
    assert result.details["sorted_hf_error"] > 0.05


def test_spectrum_fit_is_not_task_success_evidence():
    result = sft.verify.verify_core_negative_controls()
    assert result.details["spectrum_fit_corr"] > 0.99
    assert result.details["task_success"] is False
    assert result.promotion == "promote-theory-core"
