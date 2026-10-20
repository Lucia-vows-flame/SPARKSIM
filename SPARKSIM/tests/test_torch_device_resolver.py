import pytest
import torch

from MNSIM.Interface.torch_device_resolver import resolve_torch_device


def test_cpu_request_always_resolves_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_torch_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("spec", ["cuda", "cuda:0", 0, "0"])
def test_gpu_request_resolves_to_visible_cuda_device(monkeypatch, spec):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert resolve_torch_device(spec) == torch.device("cuda:0")


def test_cuda_request_preserves_explicit_device_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert resolve_torch_device("cuda:3") == torch.device("cuda:3")
    assert resolve_torch_device(2) == torch.device("cuda:2")


def test_explicit_gpu_request_fails_instead_of_falling_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_torch_device("cuda")


def test_gpu_index_out_of_range_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="out of range"):
        resolve_torch_device("cuda:2")

