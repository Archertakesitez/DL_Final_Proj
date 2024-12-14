from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from JEPA_model import JEPAModel
import glob


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JEPAModel(latent_dim=256, use_momentum=True).to(device)
    
    # Load checkpoint
    checkpoint = torch.load("model_weights.pth")
    # Extract just the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    print("Pred shape:", pred.shape)
    print("Target shape:", target.shape)
    assert pred.shape == target.shape
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    model = model.to(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
