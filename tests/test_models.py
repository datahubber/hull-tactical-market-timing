import pytest
import torch

from hull_timing.models import NEURAL_MODELS, build_model


@pytest.mark.parametrize("kind", NEURAL_MODELS)
def test_forward_backward(kind):
    torch.manual_seed(0)
    model = build_model(kind, n_features=7, seq_len=5, hidden=8, layers=2, mlp_hidden=(16,))
    x = torch.randn(4, 5, 7)
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
    torch.nn.functional.binary_cross_entropy_with_logits(out, torch.ones(4)).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_unknown_kind():
    with pytest.raises(ValueError):
        build_model("transformer", 3, 5)
