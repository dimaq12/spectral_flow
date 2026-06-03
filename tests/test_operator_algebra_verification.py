"""Independent production verification for theory-core SFT claims."""
import sft


CORE_CLAIMS = (
    "CORE-HF-001",
    "CORE-PERT-002",
    "CORE-INV-003",
    "CORE-ALG-004",
    "CORE-HESS-006",
    "CORE-TOPO-007",
    "CORE-JFUSE-008",
    "CORE-MJFUSE-009",
)


def test_core_verification_suite_has_structured_results():
    results = sft.verify.run_core_verification_suite()
    for claim_id in CORE_CLAIMS:
        row = results[claim_id]
        assert row["claim_id"] == claim_id
        assert row["layer"] == "theory-core"
        assert row["evidence_level"] in {"E3", "E4"}
        assert row["status"] == "PASS", row
        assert row["passed"] is True


def test_run_verification_suite_keeps_legacy_keys_and_adds_core_block():
    suite = sft.verify.run_verification_suite()
    assert "C1_result" in suite
    assert "G1_exceptional_point" in suite
    assert "CORE_structured" in suite
    assert suite["CORE_structured"]["CORE-HF-001"]["status"] == "PASS"


def test_core_report_helpers_are_machine_and_human_readable():
    results = sft.verify.run_core_verification_suite()
    payload = sft.verify.verification_report_jsonable(results)
    markdown = sft.verify.verification_report_markdown(results)
    assert payload["passed"] == payload["total"]
    assert "sft-independent-core-verification" == payload["suite"]
    assert "CORE-HF-001" in markdown
    assert "Promotion rule" in markdown
