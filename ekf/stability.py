"""Conservative stop conditions for sustained optimization divergence."""

import math


class ValidationEarlyStopping:
    """Patience tracks cumulative meaningful progress; checkpoint selection stays exact."""
    def __init__(self, patience=0, min_delta=1e-4):
        if patience < 0 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('Early stopping needs nonnegative patience and finite nonnegative min_delta.')
        self.patience = patience
        self.min_delta = min_delta
        self.meaningful_best = math.inf
        self.bad_epochs = 0

    def observe(self, mse):
        if not math.isfinite(mse) or mse < 0:
            raise ValueError('Early stopping requires finite nonnegative validation MSE.')
        improved = mse < self.meaningful_best and self.meaningful_best-mse >= self.min_delta
        if improved:
            self.meaningful_best = mse
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return bool(self.patience and self.bad_epochs >= self.patience)


class LossGrowthGuard:
    def __init__(self, factor=None, patience=2):
        if factor is not None and (not math.isfinite(factor) or factor <= 1):
            raise ValueError("Loss growth factor must be finite and greater than one.")
        if patience < 1:
            raise ValueError("Loss growth patience must be positive.")
        self.factor = factor
        self.patience = patience
        self.best = {}
        self.bad_epochs = {}

    def observe(self, phase, mse):
        if not math.isfinite(mse) or mse < 0:
            raise RuntimeError(f"Optimization instability: non-finite/negative {phase} MSE: {mse}")
        best = self.best.get(phase, mse)
        bad = self.factor is not None and mse > self.factor*max(best, 1e-12)
        self.bad_epochs[phase] = self.bad_epochs.get(phase, 0)+1 if bad else 0
        self.best[phase] = min(best, mse)
        if self.bad_epochs[phase] >= self.patience:
            raise RuntimeError(f"Optimization instability: {phase} MSE {mse:.6g} exceeds "
                               f"{self.factor:g}x best {best:.6g} for {self.patience} consecutive epochs.")
