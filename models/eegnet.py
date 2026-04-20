"""
EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs.

Reference: Lawhern et al., "EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain–Computer Interfaces," J. Neural Eng., 2018.

Architecture:
  Block 1: Temporal Conv → BatchNorm → Depthwise Conv → BatchNorm → ELU → AvgPool → Dropout
  Block 2: Separable Conv (Depthwise + Pointwise) → BatchNorm → ELU → AvgPool → Dropout
  Classifier: Flatten → Dense → Softmax
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet architecture for EEG classification.

    Input shape: (batch, 1, n_channels, n_timepoints)
    Output shape: (batch, n_classes)
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_timepoints: int = 256,
        n_classes: int = 4,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes

        # ----- Block 1: Temporal + Spatial Filtering -----
        # Temporal convolution: learn frequency filters
        self.temporal_conv = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise convolution: learn spatial filters per temporal filter
        self.depthwise_conv = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # ----- Block 2: Separable Convolution -----
        # Depthwise separable convolution
        self.separable_depthwise = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.separable_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # ----- Classifier -----
        # Compute flattened size dynamically
        self._flat_size = self._get_flat_size(n_channels, n_timepoints)
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self, n_channels, n_timepoints):
        """Forward a dummy tensor to compute the flattened feature size."""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self._forward_features(x)
            return x.shape[1]

    def _forward_features(self, x):
        """Extract features (everything before classifier)."""
        # Block 1
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(start_dim=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, 1, n_channels, n_timepoints)
        Returns:
            logits: (batch, n_classes)
        """
        features = self._forward_features(x)
        return self.classifier(features)

    def predict_proba(self, x):
        """Return softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def build_eegnet(config, model_type="target"):
    """
    Build EEGNet from config.

    Args:
        config: ExperimentConfig
        model_type: "target" (full model) or "substitute" (smaller variant)
    """
    mc = config.model
    dc = config.data
    if model_type == "target":
        return EEGNet(
            n_channels=dc.n_channels, n_timepoints=dc.n_timepoints,
            n_classes=dc.n_classes, F1=mc.F1, D=mc.D, F2=mc.F2,
            kernel_length=mc.kernel_length, dropout_rate=mc.dropout_rate,
        )
    elif model_type == "substitute":
        return EEGNet(
            n_channels=dc.n_channels, n_timepoints=dc.n_timepoints,
            n_classes=dc.n_classes, F1=mc.sub_F1, D=mc.sub_D, F2=mc.sub_F2,
            kernel_length=mc.sub_kernel_length, dropout_rate=mc.sub_dropout_rate,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
