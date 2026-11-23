import torch


class OutputMetrics:
    """Sample-weighted mean sqrt(variance), per sensor and x/y/z axis."""

    def __init__(self):
        self.states = {}

    @torch.no_grad()
    def _update(self, name, value):
        samples = value.detach().reshape(-1, 3).double().sqrt()
        count = samples.shape[0]
        if count == 0:
            return
        mean = samples.mean(dim=0)
        if name in self.states:
            old_count, old_mean = self.states[name]
            total = old_count + count
            mean = old_mean + (mean - old_mean) * (count / total)
            count = total
        self.states[name] = count, mean

    @torch.no_grad()
    def update(self, outputs):
        for sensor in ('gyro', 'acc'):
            variance = outputs.get(f'{sensor}_cov')
            if variance is not None:
                self._update(f'{sensor}_noise_std', variance)

    @torch.no_grad()
    def summarize(self):
        """Return six noise-level series; sqrt is applied before averaging."""
        metrics = {}
        for name, (_, mean) in self.states.items():
            for axis, scalar in zip('xyz', mean.cpu().tolist()):
                metrics[f'outputs/{name}/mean/{axis}'] = scalar
        return metrics
