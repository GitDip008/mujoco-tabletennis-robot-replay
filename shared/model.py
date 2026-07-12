import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    3-hidden-layer MLP for TTSwing stroke classification.
    Input  : (batch, 34)  — standardized IMU feature vector
    Output : (batch, 4)   — raw logits for CrossEntropyLoss
    """

    def __init__(
        self,
        input_dim: int = 34,
        hidden_dims: list[int] = [256, 128, 64],
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: dict, device: torch.device) -> MLP:
    model = MLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_dims=cfg["model"]["hidden_dims"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
    return model.to(device)