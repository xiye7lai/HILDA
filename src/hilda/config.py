from dataclasses import dataclass, field


@dataclass
class CycleCodecConfig:
    feature_dim: int = 192
    latent_dim: int = 64
    hidden_dim: int = 256
    dropout: float = 0.05
    logvar_min: float = -8.0
    logvar_max: float = 4.0


@dataclass
class LatentFlowConfig:
    latent_dim: int = 64
    max_length: int = 4904
    block_size: int = 64
    d_model: int = 128
    n_heads: int = 8
    n_history_layers: int = 2
    n_block_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.10
    terminal_block_probability: float = 0.25


@dataclass
class LossWeights:
    health: float = 2.0
    health_delta: float = 0.5
    health_monotonic: float = 0.1
    reconstruction: float = 0.25
    waveform_delta: float = 0.25
    kl: float = 1e-4
    reference: float = 0.05
    flow_waveform: float = 0.10
    critical_region: float = 0.05
    protocol_features: float = 0.02
    boundary: float = 0.05
    multiscale: float = 0.05


@dataclass
class TrainingConfig:
    flow_learning_rate: float = 3e-4
    codec_learning_rate: float = 1e-5
    flow_weight_decay: float = 0.01
    codec_weight_decay: float = 1e-4
    min_observed_fraction: float = 0.05
    max_observed_fraction: float = 0.30
    rollout_curriculum: tuple[int, ...] = (1, 2, 4, 0)
    scheduled_sampling_start: float = 0.10
    scheduled_sampling_end: float = 0.50
    rollout_flow_steps: int = 4
    flow_gradient_clip: float = 1.0
    codec_gradient_clip: float = 0.25
    weights: LossWeights = field(default_factory=LossWeights)


@dataclass
class HILDAConfig:
    codec: CycleCodecConfig = field(default_factory=CycleCodecConfig)
    flow: LatentFlowConfig = field(default_factory=LatentFlowConfig)

    def __post_init__(self) -> None:
        if self.codec.latent_dim != self.flow.latent_dim:
            raise ValueError("codec and flow latent dimensions must match")

