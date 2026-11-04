import torch


class OutputMetrics:
    """Accumulate per-axis population statistics without keeping batch tensors."""

    def __init__(self):
        self.states = {}

    @torch.no_grad()
    def _update(self, name, value):
        samples = value.detach().reshape(-1, 3).double()
        count = samples.shape[0]
        if count == 0:
            return
        variance, mean = torch.var_mean(samples, dim=0, unbiased=False)
        m2 = variance * count
        minimum, maximum = samples.amin(dim=0), samples.amax(dim=0)
        if name in self.states:
            old_count, old_mean, old_m2, old_min, old_max = self.states[name]
            total = old_count + count
            delta = mean - old_mean
            m2 = old_m2 + m2 + delta.square() * (old_count * count / total)
            mean = old_mean + delta * (count / total)
            minimum = torch.minimum(old_min, minimum)
            maximum = torch.maximum(old_max, maximum)
            count = total
        self.states[name] = count, mean, m2, minimum, maximum

    @torch.no_grad()
    def update(self, outputs):
        for name in ('correction_acc', 'correction_gyro', 'acc_cov', 'gyro_cov'):
            value = outputs.get(name)
            if value is None:
                continue
            self._update(name, value)
            if name.endswith('_cov'):
                self._update(name.replace('_cov', '_noise_std'), value.detach().double().sqrt())

    @torch.no_grad()
    def summarize(self, split, epoch=None, propcov=False):
        """Print epoch summaries and return scalar metrics for existing W&B logging."""
        epoch_label = f' epoch {epoch}' if epoch is not None else ''
        print(f'\n{split}{epoch_label} OUTPUT STATISTICS (all batch/time samples; axes x, y, z):')
        metrics = {}
        for name, (count, mean, m2, minimum, maximum) in self.states.items():
            description = name
            if name.endswith('_noise_std'):
                description += ' (predicted noise std = sqrt(variance))'
            elif name.endswith('_cov'):
                description += ' (diagonal variance)'
            print(f'   - {description}; {count} samples:')
            for stat, value in (
                ('mean', mean), ('std', (m2 / count).clamp_min(0).sqrt()),
                ('min', minimum), ('max', maximum),
            ):
                values = value.cpu().tolist()
                print(f'     {stat} = {values}')
                for axis, scalar in zip('xyz', values):
                    metrics[f'outputs/{name}/{stat}/{axis}'] = scalar
        if not propcov:
            print('   - acc_cov / gyro_cov / predicted noise std: unavailable (propcov=False)')
        return metrics
