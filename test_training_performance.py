import contextlib
import io
import unittest
from unittest import mock

import pypose as pp
import torch
from pyhocon import ConfigFactory

from model.code import CodeNet
from model.losses import get_loss
from train import _LossMetrics, train, test as validate


class ReferenceCodeNet(CodeNet):
    """Retain the old behavior: propagate covariance in both IMU passes."""

    def integrate(self, init_state, data, cov_state, use_gtrot=None, propagate_cov=None):
        return super().integrate(init_state, data, cov_state, use_gtrot=use_gtrot)


def make_batch(frames=40):
    batch = 2
    data = {
        'acc': torch.randn(batch, frames + 9, 3, dtype=torch.double),
        'gyro': torch.randn(batch, frames + 9, 3, dtype=torch.double) * 0.1,
        'dt': torch.full((batch, frames, 1), 0.005, dtype=torch.double),
        'rot': pp.randn_SO3(batch, frames, dtype=torch.double, sigma=0.01),
    }
    initial = {
        'pos': torch.randn(batch, 1, 3, dtype=torch.double),
        'vel': torch.randn(batch, 1, 3, dtype=torch.double),
        'rot': pp.randn_SO3(batch, 1, dtype=torch.double, sigma=0.1),
    }
    label = {
        'gt_pos': torch.randn(batch, frames, 3, dtype=torch.double),
        'gt_vel': torch.randn(batch, frames, 3, dtype=torch.double),
        'gt_rot': pp.randn_SO3(batch, frames, dtype=torch.double, sigma=0.1),
    }
    return data, initial, label


def make_network(conf, network_type=CodeNet):
    with contextlib.redirect_stdout(io.StringIO()):
        network = network_type(conf).double()
    network._inference_printed = network._output_printed = True
    return network


class TrainingPerformanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self):
        conf = ConfigFactory.parse_file('configs/exp/TLab/finetune_motion_body_stage2.conf').train
        conf.device = 'cpu'
        return conf

    def test_outputs_losses_and_gradients_match_with_half_the_covariance_calls(self):
        for sampling, gtrot, covaug in ((10, True, True), (False, True, True), (10, False, False)):
            with self.subTest(sampling=sampling, gtrot=gtrot, covaug=covaug):
                torch.manual_seed(17)
                conf = self.config()
                conf['sampling'], conf['gtrot'], conf['covaug'] = sampling, gtrot, covaug
                network = make_network(conf)
                # Exercise nonzero covariance gradients through the encoder too.
                for decoder in (network.acccov_decoder, network.gyrocov_decoder):
                    torch.nn.init.normal_(decoder[-1].weight, std=0.01)
                reference = make_network(conf, ReferenceCodeNet)
                reference.load_state_dict(network.state_dict(), strict=True)
                data, initial, labels = make_batch()

                rng_state = torch.get_rng_state()
                with mock.patch.object(network.integrator, 'propagate_cov', wraps=network.integrator.propagate_cov) as fast_cov:
                    actual = network(dict(data), dict(initial))
                torch.set_rng_state(rng_state)
                with mock.patch.object(reference.integrator, 'propagate_cov', wraps=reference.integrator.propagate_cov) as old_cov:
                    expected = reference(dict(data), dict(initial))
                self.assertEqual(fast_cov.call_count, 40 // sampling if sampling else 1)
                self.assertEqual(old_cov.call_count, 2 * fast_cov.call_count)
                self.assertTrue(network.integrator.prop_cov)
                self.assertEqual(actual.keys(), expected.keys())
                self.assertEqual(network.state_dict().keys(), reference.state_dict().keys())
                for key in actual:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                actual_losses = get_loss(actual, labels, conf)
                expected_losses = get_loss(expected, labels, conf)
                for key in actual_losses:
                    torch.testing.assert_close(actual_losses[key], expected_losses[key], rtol=0, atol=0)
                actual_losses['loss'].backward()
                expected_losses['loss'].backward()
                for (name, parameter), (_, original) in zip(network.named_parameters(), reference.named_parameters()):
                    with self.subTest(parameter=name):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        torch.testing.assert_close(parameter.grad, original.grad, rtol=0, atol=0)

    def test_loss_metrics_match_per_batch_scalar_averages_without_retaining_graphs(self):
        for propcov in (False, True):
            with self.subTest(propcov=propcov):
                metrics = _LossMetrics(propcov, ('vel_nll',))
                expected = {key: 0.0 for key in metrics.sources}
                for i in range(3):
                    values = {
                        key: torch.tensor([i + j + 0.25], dtype=torch.double, requires_grad=True)
                        for j, key in enumerate(metrics.sources.values())
                    }
                    metrics.update(values)
                    for name, source in metrics.sources.items():
                        expected[name] += values[source].mean().item() / 3
                self.assertFalse(metrics.total.requires_grad)
                self.assertIsNone(metrics.total.grad_fn)
                actual = metrics.averages()
                for key, value in expected.items():
                    self.assertAlmostEqual(actual[key], value)
                if not propcov:
                    self.assertEqual(actual['pred_cov_rot'], 0.0)

    def test_training_and_validation_smoke_for_both_stages(self):
        for propcov, detect_anomaly in ((False, False), (True, False), (True, True)):
            with self.subTest(propcov=propcov, detect_anomaly=detect_anomaly):
                torch.manual_seed(19)
                conf = self.config()
                conf['propcov'], conf['detect_anomaly'] = propcov, detect_anomaly
                conf['log_interval'] = 2
                network = make_network(conf)
                optimizer = torch.optim.Adam(network.parameters(), lr=conf.lr)
                before = network.accdecoder[-1].weight.detach().clone()
                batches = [make_batch() for _ in range(3)]
                anomaly_states = []
                handle = network.register_forward_pre_hook(
                    lambda *_: anomaly_states.append(torch.is_anomaly_enabled())
                )
                outer_anomaly = torch.is_anomaly_enabled()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = train(network, batches, conf, 0, optimizer)
                    handle.remove()
                    validation = validate(network, batches, conf, epoch=1)
                self.assertEqual(anomaly_states, [detect_anomaly] * 3)
                self.assertEqual(torch.is_anomaly_enabled(), outer_anomaly)
                self.assertFalse(torch.equal(before, network.accdecoder[-1].weight))
                self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
                self.assertTrue(torch.isfinite(torch.tensor(validation['loss'])))
                self.assertIn('outputs/correction_acc/mean/x', result)
                if propcov:
                    self.assertIn('vel_nll', result)
                    self.assertIn('cov_nll', validation)
                    self.assertEqual(validation['acc_covs'].numel(), 3 * 2 * 40 * 3)


if __name__ == '__main__':
    unittest.main()
