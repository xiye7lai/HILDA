from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import CycleCodecConfig, HILDAConfig, LatentFlowConfig


class CycleVAE(nn.Module):
    """MLP VAE that represents one complete cycle with one latent vector."""

    def __init__(self, config: CycleCodecConfig):
        super().__init__()
        self.config = config
        self.encoder_body = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.feature_dim),
        )

    def encode_stats(self, cycles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder_body(cycles)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(
            self.config.logvar_min, self.config.logvar_max
        )
        return mu, logvar

    def encode(self, cycles: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mu, logvar = self.encode_stats(cycles)
        if sample:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(
        self, cycles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(cycles)
        latent = mu
        if self.training:
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(latent), mu, logvar, latent

    @staticmethod
    def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = t[:, None].float() * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class BlockCausalLatentFlow(nn.Module):
    """Endpoint-prediction latent transport with detached clean history."""

    def __init__(self, config: LatentFlowConfig):
        super().__init__()
        self.config = config
        self.state_input = nn.Linear(config.latent_dim, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(1, config.max_length, config.d_model)
        )
        history_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            history_layer, config.n_history_layers
        )
        block_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.block_decoder = nn.TransformerDecoder(
            block_layer, config.n_block_layers
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )
        self.output = nn.Linear(config.d_model, config.latent_dim)
        nn.init.normal_(self.position_embedding, std=0.02)

    def encode_history(self, clean_history: torch.Tensor) -> torch.Tensor:
        """Encode visible history without propagating gradients into it."""
        length = clean_history.shape[1]
        hidden = self.state_input(clean_history.detach())
        hidden = hidden + self.position_embedding[:, :length]
        return self.history_encoder(hidden)

    def predict_x1(
        self,
        clean_history: torch.Tensor,
        noisy_block: torch.Tensor,
        t: torch.Tensor,
        block_start: int,
    ) -> torch.Tensor:
        memory = self.encode_history(clean_history)
        block_length = noisy_block.shape[1]
        block = self.state_input(noisy_block)
        block = block + self.position_embedding[
            :, block_start : block_start + block_length
        ]
        block = block + self.time_mlp(
            timestep_embedding(t, self.config.d_model)
        )[:, None, :]
        # Current-block attention is bidirectional; cross-block access is only
        # through the already finalized clean-history memory.
        return self.output(self.block_decoder(block, memory))

    @torch.no_grad()
    def generate_block(
        self, history: torch.Tensor, block_length: int, steps: int = 8
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        start = history.shape[1]
        if start + block_length > self.config.max_length:
            raise ValueError("requested block exceeds max_length")
        values = torch.randn(
            history.shape[0],
            block_length,
            self.config.latent_dim,
            device=history.device,
            dtype=history.dtype,
        )
        eps = 1e-3
        for step in range(steps):
            t_value = min(step / steps, 1.0 - eps)
            t_next = min((step + 1) / steps, 1.0 - eps)
            t = torch.full(
                (history.shape[0],),
                t_value,
                device=history.device,
                dtype=history.dtype,
            )
            endpoint = self.predict_x1(history, values, t, start)
            velocity = (endpoint - values) / max(1.0 - t_value, eps)
            values = values + (t_next - t_value) * velocity
        t = torch.full(
            (history.shape[0],),
            1.0 - eps,
            device=history.device,
            dtype=history.dtype,
        )
        return self.predict_x1(history, values, t, start)

    @torch.no_grad()
    def rollout(
        self, prefix: torch.Tensor, target_length: int, steps: int = 8
    ) -> torch.Tensor:
        if prefix.ndim != 3 or prefix.shape[-1] != self.config.latent_dim:
            raise ValueError("prefix must have shape [batch, cycles, latent_dim]")
        if not 0 < prefix.shape[1] <= target_length <= self.config.max_length:
            raise ValueError("target_length must be between prefix and max_length")
        history = prefix
        while history.shape[1] < target_length:
            count = min(
                self.config.block_size, target_length - history.shape[1]
            )
            history = torch.cat(
                [history, self.generate_block(history, count, steps)], dim=1
            )
        return history


class HILDA(nn.Module):
    """Cycle codec and block-causal latent generator in one module."""

    def __init__(self, config: HILDAConfig | None = None):
        super().__init__()
        self.config = config or HILDAConfig()
        self.codec = CycleVAE(self.config.codec)
        self.flow = BlockCausalLatentFlow(self.config.flow)
        self.register_buffer(
            "latent_mean", torch.zeros(self.config.codec.latent_dim)
        )
        self.register_buffer(
            "latent_std", torch.ones(self.config.codec.latent_dim)
        )

    def set_latent_statistics(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        if mean.shape != self.latent_mean.shape or std.shape != self.latent_std.shape:
            raise ValueError("latent statistics have the wrong shape")
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std.clamp_min(1e-6))

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latent_mean) / self.latent_std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latent_std + self.latent_mean

    @torch.no_grad()
    def generate(
        self,
        normalized_prefix_cycles: torch.Tensor,
        target_length: int,
        flow_steps: int = 8,
    ) -> torch.Tensor:
        if normalized_prefix_cycles.ndim != 3:
            raise ValueError("prefix cycles must have shape [batch, cycles, features]")
        batch, length, features = normalized_prefix_cycles.shape
        if features != self.config.codec.feature_dim:
            raise ValueError("prefix feature dimension does not match the codec")
        was_training = self.training
        self.eval()
        try:
            prefix_latents = self.codec.encode(
                normalized_prefix_cycles.reshape(-1, features)
            ).reshape(batch, length, -1)
            generated = self.flow.rollout(
                self.normalize_latents(prefix_latents), target_length, flow_steps
            )
            decoded = self.codec.decode(
                self.denormalize_latents(generated).reshape(
                    -1, self.config.codec.latent_dim
                )
            )
            decoded = decoded.reshape(batch, target_length, features)
            decoded[:, :length] = normalized_prefix_cycles
        finally:
            self.train(was_training)
        return decoded
