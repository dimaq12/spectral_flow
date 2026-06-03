"""SFT 0.3 scientific verification gates."""
import sft


def test_geometry_verification_gates():
    assert sft.verify.verify_g1_exceptional_point()
    assert sft.verify.verify_g2_biorthogonal_hf()
    assert sft.verify.verify_g3_pde_convergence() > 1.5
    g4 = sft.verify.verify_g4_second_order_inverse()
    assert g4["finite"]
    assert g4["hessian_count"] > 0


def test_run_verification_suite_includes_geometry_gates():
    result = sft.verify.run_verification_suite()
    for key in (
        "G1_exceptional_point",
        "G2_biorthogonal_hf",
        "G3_pde_convergence",
        "G4_second_order_inverse",
    ):
        assert key in result
