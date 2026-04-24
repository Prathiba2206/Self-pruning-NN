import torch


def accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def sparsity_level(model, threshold: float = 1e-2) -> float:
    gates = model.all_gates()
    if gates.numel() == 0:
        return 0.0
    return 100.0 * (gates < threshold).float().mean().item()