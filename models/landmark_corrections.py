"""Stable classification of known-negative visible surface samples.

The main rendered heatmap objective remains differentiable through geometry.
The optional auxiliary treats opacity weights as a fixed visibility selector,
so it corrects class zero without rewarding deletion of foreground opacity.
"""
import numpy as np
import torch


def nonlandmark_surface_loss(log_probabilities, weights, negative_rays,
                              min_opacity=.5, detach_visibility=True):
    if log_probabilities.shape[:-1] != weights.shape or negative_rays.shape != weights.shape[:1]:
        raise ValueError('Non-landmark logits, SDF weights and ray mask are misaligned')
    if not 0 < min_opacity <= 1:
        raise ValueError('Non-landmark opacity threshold must be in (0,1]')
    visibility = weights.detach() if detach_visibility else weights
    opacity = visibility.sum(-1)
    usable = negative_rays.bool() & (opacity.detach() >= min_opacity)
    if not usable.any():
        return log_probabilities[..., 0].sum() * 0., usable
    normalized = visibility / opacity[:, None].clamp_min(1e-8)
    # log_softmax is provided directly by the head; do not log/clamp softmax.
    per_ray = -(normalized * log_probabilities[..., 0]).sum(-1)
    return per_ray[usable].mean(), usable


def known_negative_rays(pixels, source_xy, source_valid, sigma, foreground, maximum_target=.01):
    """Missing classes are unknown; excluded visible/occluded label regions stay protected."""
    if not 0 < maximum_target < 1:
        raise ValueError('Negative target threshold must lie in (0,1)')
    if not source_valid.all():
        return torch.zeros(len(pixels), dtype=torch.bool, device=pixels.device)
    sigma = torch.as_tensor(sigma, dtype=pixels.dtype, device=pixels.device)
    distance = (pixels[:, None] - source_xy[None]).square().sum(-1)
    target = torch.exp(-distance / (2 * sigma.square()))
    return (target.max(-1).values < maximum_target) & foreground.reshape(-1).bool()


class HardNegativeCache:
    """Small per-camera cache; high-scoring known-negative foreground pixels only."""
    def __init__(self, capacity=1024):
        if capacity < 1:
            raise ValueError('Hard-negative capacity must be positive')
        self.capacity = int(capacity)
        self.pools = {}

    def get(self, image_index):
        return np.asarray(self.pools.get(str(int(image_index)), []), dtype=float).reshape(-1, 3)

    def update(self, image_index, pixels, scores, eligible):
        key = str(int(image_index))
        merged = {(int(u), int(v)): float(s) for u, v, s in self.get(image_index)}
        for (u, v), score in zip(np.asarray(pixels)[np.asarray(eligible)], np.asarray(scores)[np.asarray(eligible)]):
            merged[(int(u), int(v))] = float(score)
        ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:self.capacity]
        self.pools[key] = [[u, v, s] for (u, v), s in ranked if np.isfinite(s) and s >= 0]

    def state_dict(self):
        return {'capacity': self.capacity, 'pools': self.pools}

    def load_state_dict(self, state):
        if int(state['capacity']) != self.capacity:
            raise ValueError('Hard-negative cache capacity changed during resume')
        self.pools = {str(k): [list(row) for row in rows] for k, rows in state['pools'].items()}
