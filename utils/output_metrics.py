import torch


class OutputMetrics:
    """Accumulate statistics with bounded-memory uncertainty quantiles."""

    def __init__(self, quantile_reservoir_size=65536):
        if quantile_reservoir_size <= 0:
            raise ValueError("quantile_reservoir_size must be positive")
        self.states = {}
        self.quantile_states = {}
        self.quantile_reservoir_size = int(quantile_reservoir_size)
        self._quantile_generators = {}

    def _quantile_generator(self, device):
        key = str(device)
        if key not in self._quantile_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(0)
            self._quantile_generators[key] = generator
        return self._quantile_generators[key]

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
    def _update_quantiles(self, name, value):
        """Maintain a uniform priority reservoir without touching global RNG."""
        samples = value.detach().reshape(-1, 3).double()
        if samples.shape[0] == 0:
            return

        priorities = torch.rand(
            samples.shape[0],
            device=samples.device,
            generator=self._quantile_generator(samples.device),
        )
        limit = self.quantile_reservoir_size
        if priorities.numel() > limit:
            priorities, indices = priorities.topk(limit, sorted=False)
            samples = samples[indices]

        if name in self.quantile_states:
            old_priorities, old_samples = self.quantile_states[name]
            priorities = torch.cat((old_priorities, priorities))
            samples = torch.cat((old_samples, samples))
            if priorities.numel() > limit:
                priorities, indices = priorities.topk(limit, sorted=False)
                samples = samples[indices]

        self.quantile_states[name] = priorities, samples

    @torch.no_grad()
    def update(self, outputs):
        for name in ('correction_acc', 'correction_gyro', 'acc_cov', 'gyro_cov'):
            value = outputs.get(name)
            if value is None:
                continue
            self._update(name, value)
            if name.endswith('_cov'):
                std_name = name.replace('_cov', '_noise_std')
                noise_std = value.detach().double().clamp_min(0).sqrt()
                self._update(std_name, noise_std)
                self._update_quantiles(std_name, noise_std)

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
            if name in self.quantile_states:
                _, samples = self.quantile_states[name]
                quantiles = torch.quantile(
                    samples,
                    torch.tensor([0.5, 0.95], device=samples.device, dtype=samples.dtype),
                    dim=0,
                )
                estimate = 'estimated ' if samples.shape[0] < count else ''
                print(
                    f'     {estimate}median/p95 use a uniform reservoir of '
                    f'{samples.shape[0]} samples'
                )
                for stat, value in (('median', quantiles[0]), ('p95', quantiles[1])):
                    values = value.cpu().tolist()
                    print(f'     {stat} = {values}')
                    for axis, scalar in zip('xyz', values):
                        metrics[f'outputs/{name}/{stat}/{axis}'] = scalar
        if not propcov:
            print('   - acc_cov / gyro_cov / predicted noise std: unavailable (propcov=False)')
        return metrics
