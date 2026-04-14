import torch
import torch.nn as nn


class MaskedAttentionStatsRegressor(nn.Module):
    """Lightweight video-level head over frozen per-frame SpatialDNet features.

    The head combines attention pooling with simple masked mean/std statistics.
    This gives the model a small amount of temporal selectivity without updating
    the expensive spatial backbone.
    """

    def __init__(
        self,
        input_dim: int = 320,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.feature_norm = nn.LayerNorm(input_dim)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
        safe_mask = mask.clone()
        all_masked = safe_mask.all(dim=1)
        if all_masked.any():
            safe_mask[all_masked, 0] = False
        return safe_mask

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        frame_scores: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict video-level depression score.

        Args:
            features: Tensor shaped (B, T, C).
            mask: Bool tensor shaped (B, T), True for padding frames.
            frame_scores: Optional frame-level predictions, ignored by this head.
            return_attention: Also return normalized temporal attention weights.
        """

        mask = mask.bool()
        safe_mask = self._safe_mask(mask)
        valid = (~safe_mask).unsqueeze(-1).to(features.dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)

        norm_features = self.feature_norm(features)
        attn_logits = self.attention(norm_features).squeeze(-1)
        attn_logits = attn_logits.masked_fill(safe_mask, torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(attn_logits, dim=1).masked_fill(safe_mask, 0.0)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        attn_pooled = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)
        mean = (features * valid).sum(dim=1) / denom
        centered = (features - mean.unsqueeze(1)) * valid
        std = torch.sqrt((centered.square().sum(dim=1) / denom).clamp(min=1e-6))

        video_feat = torch.cat([attn_pooled, mean, std], dim=1)
        pred = self.regressor(video_feat)
        if return_attention:
            return pred, attn_weights
        return pred


class ResidualAttentionStatsRegressor(nn.Module):
    """Residual video-level head anchored to the frozen frame-score mean.

    The initial prediction equals the masked mean of frame-level SpatialDNet
    scores. Training can only improve the selected checkpoint if the residual
    reduces dev MAE, which makes this safer than replacing the baseline mean.
    """

    def __init__(
        self,
        input_dim: int = 320,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        residual_scale: float = 8.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.residual_scale = residual_scale
        self.feature_norm = nn.LayerNorm(input_dim)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        final = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(input_dim * 3 + 4),
            nn.Linear(input_dim * 3 + 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            final,
        )

    @staticmethod
    def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
        safe_mask = mask.clone()
        all_masked = safe_mask.all(dim=1)
        if all_masked.any():
            safe_mask[all_masked, 0] = False
        return safe_mask

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        frame_scores: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if frame_scores is None:
            raise ValueError("ResidualAttentionStatsRegressor requires frame_scores.")

        mask = mask.bool()
        safe_mask = self._safe_mask(mask)
        valid = (~safe_mask).unsqueeze(-1).to(features.dtype)
        denom = valid.sum(dim=1).clamp(min=1.0)

        norm_features = self.feature_norm(features)
        attn_logits = self.attention(norm_features).squeeze(-1)
        attn_logits = attn_logits.masked_fill(safe_mask, torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(attn_logits, dim=1).masked_fill(safe_mask, 0.0)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        attn_pooled = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)
        mean = (features * valid).sum(dim=1) / denom
        centered = (features - mean.unsqueeze(1)) * valid
        std = torch.sqrt((centered.square().sum(dim=1) / denom).clamp(min=1e-6))

        score_denom = (~safe_mask).to(frame_scores.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
        score_valid = frame_scores.masked_fill(safe_mask, 0.0)
        score_mean = score_valid.sum(dim=1, keepdim=True) / score_denom
        score_centered = (frame_scores - score_mean) * (~safe_mask).to(frame_scores.dtype)
        score_std = torch.sqrt(
            (score_centered.square().sum(dim=1, keepdim=True) / score_denom).clamp(min=1e-6)
        )
        score_max = frame_scores.masked_fill(safe_mask, torch.finfo(frame_scores.dtype).min).max(dim=1, keepdim=True).values
        score_min = frame_scores.masked_fill(safe_mask, torch.finfo(frame_scores.dtype).max).min(dim=1, keepdim=True).values
        score_stats = torch.cat([score_mean, score_std, score_max, score_min], dim=1)

        video_feat = torch.cat([attn_pooled, mean, std, score_stats], dim=1)
        residual = self.residual_scale * torch.tanh(self.residual_head(video_feat))
        pred = score_mean + residual
        if return_attention:
            return pred, attn_weights
        return pred
