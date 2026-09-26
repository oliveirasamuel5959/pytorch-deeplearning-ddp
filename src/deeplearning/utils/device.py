"""Device selection helper."""
import torch

def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve 'auto' | 'cpu' | 'cuda' | 'mps' into a concrete torch.device,
    falling back gracefully if the requested backend is unavailable."""
    if preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
