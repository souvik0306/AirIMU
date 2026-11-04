import contextlib
import io
import unittest

import torch

from utils.output_metrics import OutputMetrics


class OutputMetricsTest(unittest.TestCase):
    def summarize(self, metrics, propcov=True):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            result = metrics.summarize('TRAIN', epoch=2, propcov=propcov)
        return result, log.getvalue()

    def assert_statistics(self, result, name, samples):
        samples = samples.reshape(-1, 3).double()
        for stat, expected in (
            ('mean', samples.mean(dim=0)),
            ('std', samples.std(dim=0, unbiased=False)),
            ('min', samples.amin(dim=0)),
            ('max', samples.amax(dim=0)),
        ):
            actual = torch.tensor(
                [result[f'outputs/{name}/{stat}/{axis}'] for axis in 'xyz'],
                dtype=torch.double,
            )
            torch.testing.assert_close(actual, expected)

    def test_uneven_batches_match_full_epoch_statistics(self):
        samples = torch.arange(24, dtype=torch.double).reshape(8, 3) + 1e9
        metrics = OutputMetrics()
        metrics.update({'correction_acc': samples[:6].reshape(2, 3, 3)})
        metrics.update({'correction_acc': samples[6:].reshape(1, 2, 3)})
        result, log = self.summarize(metrics, propcov=False)
        self.assert_statistics(result, 'correction_acc', samples)
        self.assertIn('TRAIN epoch 2', log)
        self.assertIn('8 samples', log)
        self.assertIn('unavailable (propcov=False)', log)

    def test_noise_statistics_use_sqrt_of_each_variance(self):
        cov = torch.tensor([[[1., 4., 9.], [9., 16., 25.], [25., 36., 49.]]])
        metrics = OutputMetrics()
        metrics.update({'acc_cov': cov[:, :2], 'gyro_cov': cov[:, :2] / 100})
        metrics.update({'acc_cov': cov[:, 2:], 'gyro_cov': cov[:, 2:] / 100})
        result, log = self.summarize(metrics)
        self.assert_statistics(result, 'acc_cov', cov)
        self.assert_statistics(result, 'gyro_cov', cov / 100)
        self.assert_statistics(result, 'acc_noise_std', cov.double().sqrt())
        self.assert_statistics(result, 'gyro_noise_std', (cov / 100).double().sqrt())
        self.assertAlmostEqual(result['outputs/acc_noise_std/mean/x'], 3.)
        self.assertIn('predicted noise std = sqrt(variance)', log)

    def test_singleton_empty_and_missing_covariance(self):
        metrics = OutputMetrics()
        metrics.update({'correction_acc': torch.empty(1, 0, 3), 'acc_cov': None})
        metrics.update({'correction_gyro': torch.tensor([[[1., 2., 3.]]])})
        result, _ = self.summarize(metrics, propcov=False)
        self.assert_statistics(result, 'correction_gyro', torch.tensor([1., 2., 3.]))
        self.assertFalse(any('cov' in name for name in result))

    def test_accumulation_is_detached_and_each_epoch_starts_fresh(self):
        value = torch.ones(2, 4, 3, requires_grad=True)
        first_epoch = OutputMetrics()
        first_epoch.update({'correction_acc': value, 'acc_cov': value})
        for state in first_epoch.states.values():
            for tensor in state[1:]:
                self.assertFalse(tensor.requires_grad)
                self.assertIsNone(tensor.grad_fn)
        second_epoch = OutputMetrics()
        second_epoch.update({'correction_acc': value * 4})
        result, _ = self.summarize(second_epoch)
        self.assertEqual(result['outputs/correction_acc/mean/x'], 4.)
        self.assertTrue(all(isinstance(value, float) for value in result.values()))


if __name__ == '__main__':
    unittest.main()
