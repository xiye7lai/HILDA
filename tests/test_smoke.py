import torch
import torch.nn as nn

from hilda import (
    CycleCodecConfig,
    HILDA,
    HILDAConfig,
    HILDATrainer,
    LatentFlowConfig,
    TrainingConfig,
)


class TinySOHEstimator(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, cycles: torch.Tensor) -> torch.Tensor:
        return self.network(cycles).squeeze(-1)


def tiny_model() -> HILDA:
    return HILDA(
        HILDAConfig(
            codec=CycleCodecConfig(
                feature_dim=12,
                latent_dim=4,
                hidden_dim=16,
                dropout=0.0,
            ),
            flow=LatentFlowConfig(
                latent_dim=4,
                max_length=12,
                block_size=4,
                d_model=16,
                n_heads=4,
                n_history_layers=1,
                n_block_layers=1,
                d_ff=32,
                dropout=0.0,
            ),
        )
    )


def test_generation_shape_and_prefix_preservation():
    torch.manual_seed(7)
    model = tiny_model().eval()
    prefix = torch.randn(2, 3, 12)
    result = model.generate(prefix, target_length=9, flow_steps=2)
    assert result.shape == (2, 9, 12)
    torch.testing.assert_close(result[:, :3], prefix, check_stride=False)


def test_training_keeps_guides_frozen():
    torch.manual_seed(11)
    model = tiny_model()
    estimator = TinySOHEstimator(12)
    trainer = HILDATrainer(
        model=model,
        reference_codec=None,
        functional_estimator=estimator,
        waveform_center=torch.zeros(12),
        waveform_scale=torch.ones(12),
        config=TrainingConfig(
            rollout_curriculum=(1,),
            scheduled_sampling_start=0.0,
            scheduled_sampling_end=0.0,
            rollout_flow_steps=1,
        ),
    )
    waveforms = torch.randn(2, 8, 12)
    soh = torch.linspace(1.0, 0.8, 8).repeat(2, 1)
    metrics = trainer.train_step(
        waveforms,
        soh,
        torch.tensor([8, 7]),
        step=1,
        total_steps=1,
    )
    assert torch.isfinite(torch.tensor(metrics["total"]))
    assert all(p.grad is None for p in trainer.reference_codec.parameters())
    assert all(p.grad is None for p in estimator.parameters())
