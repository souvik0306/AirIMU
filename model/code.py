import torch
import numpy as np
import pypose as pp
import math

import torch.nn as nn
import torch.nn.functional as F
from model.net import ModelBase
from model.cnn import CNNEncoder


@torch.no_grad()
def _print_output_summary(name, value, variance=False):
    """Preview the first sequence and summarize all batch/time samples by axis."""
    value = value.detach().cpu()
    print(f"   - {name} shape: {value.shape}")
    if value.numel() == 0:
        print(f"   - {name}: no samples")
        return

    print(f"   - First {min(3, value.shape[1])} {name} values:")
    for i in range(min(3, value.shape[1])):
        print(f"     {name}[{i}] = {value[0, i, :].tolist()}")

    samples = value.reshape(-1, value.shape[-1])
    print(f"   - {name} statistics (all batch/time samples; axes x, y, z):")
    for label, stat in (
        ("mean", samples.mean(dim=0)),
        ("std", samples.std(dim=0, unbiased=False)),
        ("min", samples.amin(dim=0)),
        ("max", samples.amax(dim=0)),
    ):
        print(f"     {label} = {stat.tolist()}")
    if variance:
        print(f"     mean predicted noise std (sqrt(variance)) = {samples.sqrt().mean(dim=0).tolist()}")


class CodeNet(ModelBase):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        gyro_std = np.pi/180
        if "gyro_std" in conf:
            print(" The gyro std is set to ", conf.gyro_std, " rad/s")
            gyro_std = conf.gyro_std
        self.register_buffer('gyro_std', torch.tensor(gyro_std))

        acc_std = 0.1
        if "acc_std" in conf:
            print(" The acc std is set to ", conf.acc_std, " m/s^2")
            acc_std = conf.acc_std
        self.register_buffer('acc_std', torch.tensor(acc_std))

        # Stage 2 predicts an unbounded positive variance using softplus. The
        # epsilon is only a numerical floor for Cholesky/NLL stability; it is
        # deliberately not a learned-uncertainty ceiling.
        covariance_epsilon = float(conf.get("covariance_epsilon", 1e-6))
        if covariance_epsilon <= 0:
            raise ValueError(
                f"covariance_epsilon must be positive, got {covariance_epsilon}."
            )
        self.register_buffer(
            'covariance_epsilon',
            torch.tensor(covariance_epsilon),
            persistent=False,
        )
        self.acc_cov_target = float(conf.get("acc_cov_target", 0.07 ** 2))
        self.gyro_cov_target = float(conf.get("gyro_cov_target", 0.003 ** 2))
        self.use_initial_sensor_bias = bool(
            conf.get("use_initial_sensor_bias", False)
        )

        ## the encoder have the same correction in one interval 
        self.interval = 9
        self.inter_head = np.floor(self.interval/2.).astype(int)
        self.inter_tail = self.interval - self.inter_head

        self.cnn = CNNEncoder(c_list=[6, 32, 64], k_list=[7, 7], s_list=[3, 3])# (N,F/8,64)

        self.gru1 = nn.GRU(input_size = 64, hidden_size = 128, num_layers = 1, batch_first = True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size = 256, num_layers = 1, batch_first = True)

        self.accdecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.acccov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

        self.gyrodecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.gyrocov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

        # Lets the covariance heads scale their output by how much bias the
        # correction network thinks it just removed, without ever feeding the
        # correction itself downstream (the deployed EKF only consumes raw
        # acc/gyro + this covariance). Zero-initialized so Stage 2 starts at
        # the configured softplus target and only learns to use this signal if
        # it actually helps the NLL.
        self.acc_bias_gain = nn.Linear(3, 3, bias=False)
        self.gyro_bias_gain = nn.Linear(3, 3, bias=False)

        self._stage2_frozen_modules = ()
        self.reset_cov_heads()

    def reset_cov_heads(self):
        """Reset every trainable uncertainty-head parameter.

        Stage 2 calls this *after* loading Stage 1. Resetting only the final
        projection would retain hidden covariance features from Stage 1 and
        would therefore not be a clean uncertainty-head restart.
        """
        epsilon = float(self.covariance_epsilon)
        for name, decoder, target in (
            ("acc_cov_target", self.acccov_decoder, self.acc_cov_target),
            ("gyro_cov_target", self.gyrocov_decoder, self.gyro_cov_target),
        ):
            softplus_target = float(target) - epsilon
            if softplus_target <= 0:
                raise ValueError(
                    f"{name} must be greater than covariance_epsilon "
                    f"({epsilon}), got {target}."
                )
            for layer in decoder:
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()
            last_linear = decoder[-1]
            nn.init.zeros_(last_linear.weight)
            # Stable inverse of softplus(y) = log(1 + exp(x)).
            init_bias = softplus_target + math.log(-math.expm1(-softplus_target))
            nn.init.constant_(last_linear.bias, init_bias)

        # These layers are part of the uncertainty prediction path, not the
        # correction path. Zero initialization makes a fresh Stage 2 begin at
        # the configured variance target without correction-dependent scaling.
        nn.init.zeros_(self.acc_bias_gain.weight)
        nn.init.zeros_(self.gyro_bias_gain.weight)

    def configure_stage2_training(self):
        """Freeze Stage 1 and expose only uncertainty-head parameters."""
        uncertainty_modules = (
            self.acccov_decoder,
            self.gyrocov_decoder,
            self.acc_bias_gain,
            self.gyro_bias_gain,
        )
        frozen_modules = (
            self.cnn,
            self.gru1,
            self.gru2,
            self.accdecoder,
            self.gyrodecoder,
        )

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in uncertainty_modules:
            module.requires_grad_(True)

        self._stage2_frozen_modules = frozen_modules
        for module in frozen_modules:
            module.eval()

        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train(self, mode=True):
        """Keep a frozen Stage 1 backbone deterministic during Stage 2."""
        super().train(mode)
        if mode:
            for module in self._stage2_frozen_modules:
                module.eval()
        return self

    def encoder(self, x):
        x = self.cnn(x.transpose(-1,-2)).transpose(-1,-2)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        return x

    def cov_decoder(self, x, correction):
        # correction is detached: it only informs the covariance heads of the
        # estimated bias magnitude, it must not let the covariance loss steer
        # the correction network (that is the state loss's job alone).
        correction = correction.detach()
        acc_raw = self.acccov_decoder(x) + self.acc_bias_gain(correction[..., :3])
        gyro_raw = self.gyrocov_decoder(x) + self.gyro_bias_gain(correction[..., 3:])

        # Positive, numerically safe and intentionally unbounded above.
        acc = F.softplus(acc_raw) + self.covariance_epsilon
        gyro = F.softplus(gyro_raw) + self.covariance_epsilon

        return torch.cat([acc, gyro], dim = -1)

    def decoder(self, x):
        acc = self.accdecoder(x) * self.acc_std
        gyro = self.gyrodecoder(x) * self.gyro_std

        return torch.cat([acc, gyro], dim = -1)

    def bias_correct_raw_imu(self, data):
        """Subtract the flight's fixed EKF2 bias without changing network input."""
        required = ("initial_acc_bias", "initial_gyro_bias")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                "Stage 2 initial bias subtraction is enabled, but the batch is "
                f"missing {missing}."
            )

        acc_bias = data["initial_acc_bias"]
        gyro_bias = data["initial_gyro_bias"]
        batch_size = data["acc"].shape[0]
        expected_shape = (batch_size, 3)
        if (
            tuple(acc_bias.shape) != expected_shape
            or tuple(gyro_bias.shape) != expected_shape
        ):
            raise ValueError(
                "Initial sensor biases must both have shape [batch, 3]; got "
                f"acc={tuple(acc_bias.shape)}, gyro={tuple(gyro_bias.shape)}."
            )

        # Skip the synthetic CNN context exactly as for the unmodified Stage 2
        # integration. The encoder above still consumes the original raw IMU.
        acc = data['acc'][:, self.interval:, :] - acc_bias[:, None, :]
        gyro = data['gyro'][:, self.interval:, :] - gyro_bias[:, None, :]
        return acc, gyro

    def _update(self, to_update, feat, frame_len):
        """Scatter corrections over the original sequence length.

        Parameters
        ----------
        to_update : torch.Tensor
            Tensor to accumulate the corrections into.
        feat : torch.Tensor
            Predicted correction features.
        frame_len : int or Tensor
            Length of the original sequence before padding.
        """

        # ``frame_len`` may be a ``torch.SymInt``/Tensor during ONNX tracing.
        # Convert to a Python integer so that math utilities work consistently.
        if not isinstance(frame_len, int):
            frame_len = int(frame_len.item() if hasattr(frame_len, "item") else frame_len)

        def _clip(x, l):
            if x > l:
                return l
            if x < 0:
                return 0
            return x

        _feat_range = math.ceil((frame_len - self.inter_head) / self.interval) + 1

        for i in range(_feat_range):
            s_p = _clip(i * self.interval - self.inter_head, frame_len)
            e_p = _clip(i * self.interval + self.inter_tail, frame_len)
            idx = _clip(i, feat.shape[1] - 1)

            # skip the first padded input
            to_update[:, s_p:e_p, :] += feat[:, idx:idx + 1, :]

        return to_update

    def inference(self, data):
        frame_len = data["acc"].shape[1] - self.interval
        
        # Debug print for first call
        if not hasattr(self, '_inference_printed'):
            print("\n" + "="*80)
            print("MODEL INFERENCE - DETAILED BREAKDOWN")
            print("="*80)
            
            print(f"\n5. MODEL INPUT (after padding):")
            print(f"   - Input acc shape: {data['acc'].shape}")
            print(f"   - Input gyro shape: {data['gyro'].shape}")
            print(f"   - self.interval (frames to skip): {self.interval}")
            print(f"   - frame_len (output length): {frame_len}")
            print(f"   - Calculation: frame_len = {data['acc'].shape[1]} - {self.interval} = {frame_len}")
            
            print(f"\n6. MODEL PROCESSING:")
            print(f"   - The model will process ALL {data['acc'].shape[1]} input frames")
            print(f"   - But only output corrections for the last {frame_len} frames")
            print(f"   - This means: skip first {self.interval} padding frames")
            
            self._inference_printed = True
        
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature)[:,1:,:]
        correction = self.decoder(feature)
        zero_signal = torch.zeros_like(data['acc'][:,self.interval:,:])

        # a referenced size 1000
        correction_acc = self._update(zero_signal.clone(), correction[...,:3], frame_len)
        correction_gyro = self._update(zero_signal.clone(), correction[...,3:], frame_len)
        
        # covariance propagation
        cov_state = {'acc_cov':None, 'gyro_cov': None,}
        if self.conf.propcov:
            cov = self.cov_decoder(feature, correction)
            cov_state['acc_cov'] = self._update(torch.zeros_like(correction_acc, device=correction_acc.device),
                                                cov[...,:3], frame_len)
            cov_state['gyro_cov'] = self._update(torch.zeros_like(correction_gyro, device=correction_gyro.device),
                                                cov[...,3:], frame_len)

        # Print once, after covariance decoding so all outputs are available.
        if not hasattr(self, '_output_printed'):
            print(f"\n7. MODEL OUTPUT:")
            print(f"   - Output length matches real data: {frame_len} frames")
            print("   - Sample previews use batch item 0")
            _print_output_summary('correction_acc', correction_acc)
            _print_output_summary('correction_gyro', correction_gyro)
            if self.conf.propcov:
                print("   - Covariance outputs are diagonal variances (axes x, y, z)")
                _print_output_summary('acc_cov', cov_state['acc_cov'], variance=True)
                _print_output_summary('gyro_cov', cov_state['gyro_cov'], variance=True)
            else:
                print("   - acc_cov / gyro_cov: unavailable (propcov=False; covariance prediction disabled)")
            print("="*80 + "\n")
            self._output_printed = True
        
        return {"cov_state": cov_state, 'correction_acc': correction_acc, 'correction_gyro': correction_gyro}

    def forward(self, data, init_state):
        inference_state = self.inference(data)

        data['corrected_acc'] = data['acc'][:,self.interval:,:] + inference_state['correction_acc']
        data['corrected_gyro'] = data['gyro'][:,self.interval:,:] + inference_state['correction_gyro']

        is_stage2 = self.conf.get('stage', 1) == 2

        # Stage 1 learns a corrected trajectory and the uncertainty of that
        # same trajectory. Ground-truth orientation is used only through the
        # supplied initial state; every subsequent orientation is propagated
        # from the corrected gyroscope.
        out_state = self.integrate(
            init_state=init_state, data=data,
            cov_state=inference_state['cov_state'],
            use_gtrot=False if not is_stage2 else None,
            propagate_cov=self.conf.propcov if not is_stage2 else False,
        )

        result = {
            'pos': out_state['pos'],
            'vel': out_state['vel'],
            'rot': out_state['rot'],
            'correction_acc': inference_state['correction_acc'],
            'correction_gyro': inference_state['correction_gyro'],
            'corrected_acc': data['corrected_acc'],
            'corrected_gyro': data['corrected_gyro'],
        }

        if self.conf.propcov and not is_stage2:
            result['cov'] = out_state['cov']
            result['acc_cov'] = inference_state['cov_state']['acc_cov']
            result['gyro_cov'] = inference_state['cov_state']['gyro_cov']

        if self.conf.propcov and is_stage2:
            # Covariance is supervised against raw IMU after subtracting the one
            # fixed EKF2 initial bias for this flight. The encoder still receives
            # untouched raw IMU, and learned AirIMU corrections are not applied to
            # this integration target. The correction knowledge instead reaches the
            # covariance heads through cov_decoder's acc/gyro_bias_gain (see
            # inference()), which conditions the predicted variance on the
            # detected bias magnitude without altering what's fed downstream.
            # use_gtrot=False (own rotation estimate, no ground-truth gravity
            # compensation) matches the deployment condition (gtrot is forced off
            # in export/inference), so the calibrated uncertainty stays valid
            # outside of training.
            raw_data = dict(data)
            if self.use_initial_sensor_bias:
                raw_acc, raw_gyro = self.bias_correct_raw_imu(data)
            else:
                raw_acc = data['acc'][:, self.interval:, :]
                raw_gyro = data['gyro'][:, self.interval:, :]
            raw_data['corrected_acc'] = raw_acc
            raw_data['corrected_gyro'] = raw_gyro
            raw_state = self.integrate(
                init_state=init_state,
                data=raw_data,
                cov_state=inference_state['cov_state'],
                use_gtrot=False,
            )

            result['raw_pos'] = raw_state['pos']
            result['raw_vel'] = raw_state['vel']
            result['raw_rot'] = raw_state['rot']
            result['cov'] = raw_state['cov']
            result['acc_cov'] = inference_state['cov_state']['acc_cov']
            result['gyro_cov'] = inference_state['cov_state']['gyro_cov']

        return result


class CodePoseNet(CodeNet):
    def __init__(self, conf):
        super().__init__(conf)

    def inference(self, data):
        frame_len = data["acc"].shape[1] - self.interval
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature)[:,1:,:]
        correction = self.decoder(feature)
        zero_signal = torch.zeros_like(data['acc'][:,self.interval:,:])

        # a referenced size 1000
        correction_acc = self._update(zero_signal.clone(), correction[...,:3], frame_len)
        correction_gyro = zero_signal.clone()

        # covariance propagation
        cov_state = {'acc_cov':None, 'gyro_cov': None,}
        if self.conf.propcov:
            cov = self.cov_decoder(feature, correction)
            cov_state['acc_cov'] = self._update(torch.zeros_like(correction_acc, device=correction_acc.device),
                                                cov[...,:3], frame_len)
            cov_state['gyro_cov'] = self._update(torch.zeros_like(correction_gyro, device=correction_gyro.device),
                                                cov[...,3:], frame_len)
        
        return {"cov_state": cov_state, 'correction_acc': correction_acc, 'correction_gyro': correction_gyro}


class CodeNetKITTI(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.integrator = pp.module.IMUPreintegrator(prop_cov=conf.propcov, reset=True).double()
        
        self.accEncoder  = CNNEncoder(k_list=[7, 3, 3], p_list=[3, 1, 1], c_list=[3, 32, 64, 128])
        self.gyroEncoder = CNNEncoder(k_list=[7, 3, 3], p_list=[3, 1, 1], c_list=[3, 32, 64, 128])
        
        self.accDecoder = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3)
        )
        self.gyroDecoder = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3)
        )
        self.accCovDecoder  = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 3)
        )
        self.gyroCovDecoder = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 3)
        )

        gyro_std = np.pi/180
        self.register_buffer('gyro_std', torch.tensor(gyro_std))

        acc_std = 0.1
        self.register_buffer('acc_std', torch.tensor(acc_std))
    
    def integrate(self, init_state, data, cov_state, use_gtrot):
        gt_rot = None
        if self.conf.gtrot: gt_rot = data['rot'].double()
        if not use_gtrot: gt_rot = None

        if self.conf.propcov:
            out_state = self.integrator(
                init_state = init_state, 
                dt = data['dt'].double(), 
                gyro = data['corrected_gyro'].double(),
                acc = data['corrected_acc'].double(), 
                rot = gt_rot, 
                acc_cov = cov_state['acc_cov'].double(), 
                gyro_cov = cov_state['gyro_cov'].double()
            )
        else:
            out_state = self.integrator(
                init_state = init_state, 
                dt = data['dt'].double(), 
                gyro = data['corrected_gyro'].double(),
                acc = data['corrected_acc'].double(), 
                rot = gt_rot, 
            )
        
        return {**out_state, **cov_state}

    def inference(self, data):
        feature_acc  = self.accEncoder(data["acc"].transpose(-1,-2)).transpose(-1,-2)
        feature_gyro = self.gyroEncoder(data["gyro"].transpose(-1,-2)).transpose(-1,-2)
        
        correction_acc  = self.accDecoder(feature_acc)
        correction_gyro = self.gyroDecoder(feature_gyro)

        cov_state = {'acc_cov':None, 'gyro_cov': None}
        if self.conf.propcov:
            feature = torch.cat([feature_acc, feature_gyro], dim = -1)
            cov_state['acc_cov']  = self.accCovDecoder(feature).exp()
            cov_state['gyro_cov'] = self.gyroCovDecoder(feature).exp()
        
        return {"cov_state": cov_state, 'correction_acc': correction_acc, 'correction_gyro': correction_gyro}

    def forward(self, data, init_state, use_gtrot=True):   
        init_state_ = {
            "pos": init_state["pos"],
            "rot": init_state["rot"][:,:1,:],
            "vel": init_state["vel"],
        }     
        inference_state = self.inference(data)

        data['corrected_acc'] = data['acc'] + inference_state['correction_acc']
        data['corrected_gyro'] = data['gyro'] + inference_state['correction_gyro']

        out_state = self.integrate(init_state=init_state_, data = data, cov_state = inference_state['cov_state'], use_gtrot=use_gtrot)
        
        return {
            **out_state, 
            'correction_acc': inference_state['correction_acc'], 
            'correction_gyro': inference_state['correction_gyro'], 
            'corrected_acc': data['corrected_acc'], 
            'corrected_gyro': data['corrected_gyro']
        }
