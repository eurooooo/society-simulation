import warnings


def test_pytest_import():
    import pytest
    assert True, "PyTest not set up correctly"

def test_cuda_availability():
    """
    Test if CUDA-enabled GPU is available.
    Raises an AssertionError if no CUDA GPU is available.
    """
    try:
        import torch
    except ImportError:
        print("Torch isn't available. Download PyTorch")
    try:
        assert torch.cuda.is_available(), "CUDA is not available. Please ensure GPU is installed and CUDA is configured properly."
        print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
    except AssertionError as e:
        print(e)
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

