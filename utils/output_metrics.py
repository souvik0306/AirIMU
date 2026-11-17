import torch


class OutputMetrics:
    """Accumulate only x/y population std of predicted sensor noise std."""

    def __init__(self):
        self.states = {}

    @torch.no_grad()
    def _update(self, name, value):
        samples = value.detach().reshape(-1, 3)[:, :2].double().sqrt()
        count = samples.shape[0]
        if count == 0:
            return
        variance, mean = torch.var_mean(samples, dim=0, unbiased=False)
        m2 = variance * count
        if name in self.states:
            old_count, old_mean, old_m2 = self.states[name]
            total = old_count + count
            delta = mean - old_mean
            m2 = old_m2 + m2 + delta.square() * (old_count * count / total)
            mean = old_mean + delta * (count / total)
            count = total
        self.states[name] = count, mean, m2

    @torch.no_grad()
    def update(self, outputs):
        for sensor in ('gyro', 'acc'):
            variance = outputs.get(f'{sensor}_cov')
            if variance is not None:
                self._update(f'{sensor}_noise_std', variance)

    @torch.no_grad()
    def summarize(self):
        """Return the four noise-spread series for the shared metric reporter."""
        metrics = {}
        for name, (count, _, m2) in self.states.items():
            values = (m2 / count).clamp_min(0).sqrt().cpu().tolist()
            for axis, scalar in zip('xy', values):
                metrics[f'outputs/{name}/std/{axis}'] = scalar
        return metrics
