"""Loss and full-history gradient parity for the compiled aligned replay."""
import unittest
import torch
from ekf import DifferentiableESKF, ViconEvent
from ekf.compiled import velocity_sequence


torch.set_num_threads(1)


def fixture():
    ekf = DifferentiableESKF()
    state, P = ekf.initialize([.1, -.2, .3], [.2, -.1, .05], [1., .02, -.01, .03],
                              [.01, -.02, .03], [.001, -.002, .003])
    times = torch.tensor([0., .005, .011, .016, .023, .028, .034, .041], dtype=torch.float64)
    imu = torch.tensor([.3, -.2, 9.9, .02, -.03, .1], dtype=torch.float64).repeat(len(times), 1)
    events = []
    for t in times[1:4].tolist():
        events.extend((ViconEvent(t, 'position', torch.tensor([.1+.2*t, -.2-.1*t], dtype=torch.float64)),
                       ViconEvent(t, 'velocity_xy', torch.tensor([.2, -.1], dtype=torch.float64)),
                       ViconEvent(t, 'yaw', torch.tensor(.06+.01*t, dtype=torch.float64))))
    flight = dict(raw_acceleration=imu[:, :3], raw_gyroscope=imu[:, 3:], imu_time=times,
                  vicon_events=events, cutoff_timestamp=.023, initial_state=state, initial_covariance=P)
    return ekf, flight, imu


class CompiledESKFTests(unittest.TestCase):
    def check_parity(self, device):
        ekf, flight, imu = fixture()
        ekf.to(device)
        reference_variance = torch.linspace(.0001, .006, len(imu)*6, dtype=torch.float64,
                                            device=device).reshape(-1, 6).requires_grad_()
        reference = ekf.run_sequence(imu, flight['imu_time'], reference_variance,
            flight['vicon_events'], flight['cutoff_timestamp'], flight['initial_state'],
            flight['initial_covariance'], return_covariance=False)['velocity']
        weights = torch.arange(len(imu)*3, dtype=torch.float64, device=device).reshape(-1, 3)
        gradient, = torch.autograd.grad((reference*weights).sum(), reference_variance)
        variance = reference_variance.detach().clone().requires_grad_()
        actual = velocity_sequence(ekf, flight, variance)
        actual_gradient, = torch.autograd.grad((actual*weights).sum(), variance)
        torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-9)
        torch.testing.assert_close(actual_gradient, gradient, atol=1e-10, rtol=1e-7)
        self.assertGreater(float(actual_gradient[:3].norm()), 0.)
        torch.testing.assert_close(actual_gradient[4:], torch.zeros_like(actual_gradient[4:]), atol=0, rtol=0)
        with torch.no_grad():
            validation = velocity_sequence(ekf, flight, variance.detach())
        torch.testing.assert_close(validation, reference, atol=1e-11, rtol=1e-9)

    def test_cpu_full_history_and_validation(self):
        self.check_parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_full_history_and_validation(self):
        self.check_parity('cuda')

    def test_rejects_asynchronous_schedule(self):
        ekf, flight, imu = fixture()
        flight['vicon_events'][0] = ViconEvent(.001, 'position', torch.zeros(2, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, 'aligned'):
            velocity_sequence(ekf, flight, torch.ones(len(imu), 6, dtype=torch.float64))

    def test_rejects_invalid_variance(self):
        ekf, flight, imu = fixture()
        for values in (-torch.ones(len(imu), 6, dtype=torch.float64),
                       torch.ones(len(imu), 6, dtype=torch.float32)):
            with self.assertRaises(ValueError):
                velocity_sequence(ekf, flight, values)


if __name__ == '__main__':
    unittest.main()
