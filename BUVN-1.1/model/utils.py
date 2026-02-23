import torch

def get_device() -> str:
    """Helper to get the optimal device available."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # useful for testing on apple silicon
        return "mps"
    return "cpu"

def print_model_parameters(model: torch.nn.Module):
    """Prints the number of trainable parameters in the model."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
    return num_params
