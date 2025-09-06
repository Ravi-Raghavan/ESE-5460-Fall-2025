from problem3 import linear_t
import numpy as np

def test_forward_computation_matches_manual():
    layer = linear_t()
    hl = np.random.randn(3, 392)
    out1 = layer.forward(hl)
    out2 = hl @ layer.w.T + layer.b
    np.testing.assert_allclose(out1, out2, rtol=1e-6, atol=1e-6)

# test_forward_computation_matches_manual()

def test_backward_computation_matches_manual():
    layer = linear_t()
    hl = np.random.randn(2, 392)
    layer.forward(hl)
    dhl_plus_1 = np.random.randn(2, 10)

    # Run backward
    dhl = layer.backward(dhl_plus_1)

    # Manual computations
    dhl_manual = dhl_plus_1 @ layer.w
    dw_manual = dhl_plus_1.T @ hl
    db_manual = dhl_plus_1.sum(axis=0)

    np.testing.assert_allclose(dhl, dhl_manual, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(layer.dw, dw_manual, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(layer.db, db_manual, rtol=1e-6, atol=1e-6)

test_backward_computation_matches_manual
