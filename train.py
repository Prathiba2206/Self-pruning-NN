import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from models import SelfPruningMLP
from utils import accuracy, sparsity_level


def get_dataloaders(data_dir: str = "./data", batch_size: int = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    # use a subset of training data to speed up on CPU
    subset_size = 30000  
    val_size = len(full_train) - subset_size
    train_set, _ = random_split(
        full_train,
        [subset_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,      # safer on Windows
        pin_memory=False,   # no GPU, so disable pinned memory
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, test_loader


def plot_gates(gates, out_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=50, color="#4C78A8", edgecolor="white")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.title("Final Gate Value Distribution (Best Model)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_two_phase(
    lam_phase1: float,
    lam_phase2: float,
    epochs_phase1: int,
    epochs_phase2: int,
    train_loader,
    test_loader,
    device,
    lr: float = 1e-3,
):
    model = SelfPruningMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_gates = sum(layer.gate_scores.numel() for layer in model.prunable_layers())
    num_gates = max(num_gates, 1)

    def run_phase(lam: float, epochs: int):
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                cls_loss = criterion(logits, y)
                sparse_loss = model.sparsity_loss()
                loss = cls_loss + lam * (sparse_loss / num_gates)
                loss.backward()
                opt.step()
            # print small log per epoch
            print(f"  Epoch done | lambda={lam:.1e} | cls_loss={cls_loss.item():.4f}")

    # Phase 1 – warm-up (small λ)
    run_phase(lam_phase1, epochs_phase1)
    # Phase 2 – pruning (larger λ)
    run_phase(lam_phase2, epochs_phase2)

    test_acc = accuracy(model, test_loader, device)
    spar = sparsity_level(model)
    gates = model.all_gates().cpu().numpy()
    return model, test_acc, spar, gates


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    train_loader, test_loader = get_dataloaders()

    # (phase1 λ, phase2 λ) configs
    lambda_configs = [
    (0.0, 0.0),      # baseline: no pruning
    (1e-4, 1e-4),
    (1e-4, 1e-3),
    (1e-4, 1e-2),
]

    results = []
    best_acc = -1.0
    best_gates = None

    for lam1, lam2 in lambda_configs:
        print(f"Training with two-phase schedule: phase1={lam1}, phase2={lam2}")
        model, test_acc, spar, gates = train_two_phase(
            lam_phase1=lam1,
            lam_phase2=lam2,
            epochs_phase1=1,   # was 3, reduced for speed
            epochs_phase2=1,   # was 3, reduced for speed
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            lr=1e-3,
        )
        results.append(
            {
                "lambda_phase1": lam1,
                "lambda_phase2": lam2,
                "test_accuracy": float(test_acc),
                "sparsity": float(spar),
            }
        )
        if test_acc > best_acc:
            best_acc = test_acc
            best_gates = gates

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if best_gates is not None:
        plot_gates(best_gates, out_dir / "gate_distribution.png")

    print("Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()