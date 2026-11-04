import torch
import numpy as np
import pypose as pp
import math

import torch.nn as nn
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

        # Hard ceiling on predicted variance so the covariance head cannot
        # "cheat" the NLL loss by inflating uncertainty without limit.
        # cov_max is the original datasheet noise density, squared (never to
        # be exceeded); cov_target is the desired operating point (1/5 of the
        # datasheet std) used only to seed initialization and as a soft
        # regularization anchor in losses.py — the model can still use the
        # full range up to cov_max when the data genuinely demands it.
        # Non-persistent: these are config-derived constants, not learned
        # state, so they must never appear in / be restored from a checkpoint
        # (older checkpoints predate these keys and would fail a strict load).
        acc_cov_min = conf.get("acc_cov_min", 0.01 ** 2)
        acc_cov_max = conf.get("acc_cov_max", 0.35 ** 2)
        self.acc_cov_target = conf.get("acc_cov_target", 0.07 ** 2)
        gyro_cov_min = conf.get("gyro_cov_min", 0.0001 ** 2)
        gyro_cov_max = conf.get("gyro_cov_max", 0.015 ** 2)
        self.gyro_cov_target = conf.get("gyro_cov_target", 0.003 ** 2)
        self.register_buffer('acc_cov_min', torch.tensor(acc_cov_min), persistent=False)
        self.register_buffer('acc_cov_max', torch.tensor(acc_cov_max), persistent=False)
        self.register_buffer('gyro_cov_min', torch.tensor(gyro_cov_min), persistent=False)
        self.register_buffer('gyro_cov_max', torch.tensor(gyro_cov_max), persistent=False)

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

        self.reset_cov_heads()

    def reset_cov_heads(self):
        """(Re-)seed the covariance heads at the desired target (not the
        sigmoid midpoint), so training has to justify raising uncertainty
        above it. Must be called again after loading a pretrained checkpoint:
        those heads are dead weight carried over from a propcov=False stage
        and would otherwise silently overwrite this initialization.
        """
        conf = self.conf
        acc_default_fraction = (self.acc_cov_target - self.acc_cov_min) / (self.acc_cov_max - self.acc_cov_min)
        gyro_default_fraction = (self.gyro_cov_target - self.gyro_cov_min) / (self.gyro_cov_max - self.gyro_cov_min)
        acc_init_fraction = conf.get("acc_cov_init_fraction", conf.get("cov_init_fraction", acc_default_fraction))
        gyro_init_fraction = conf.get("gyro_cov_init_fraction", conf.get("cov_init_fraction", gyro_default_fraction))
        for decoder, init_fraction in (
            (self.acccov_decoder, acc_init_fraction),
            (self.gyrocov_decoder, gyro_init_fraction),
        ):
            last_linear = decoder[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, math.log(float(init_fraction) / (1 - float(init_fraction))))

    def encoder(self, x):
        x = self.cnn(x.transpose(-1,-2)).transpose(-1,-2)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        return x

    def cov_decoder(self, x):
        # Bounded sigmoid parameterization: variance is always within
        # [cov_min, cov_max] regardless of the raw network output, so the
        # model cannot drive uncertainty above the sensor's physical noise
        # floor to minimize the NLL loss.
        acc = self.acc_cov_min + (self.acc_cov_max - self.acc_cov_min) * torch.sigmoid(self.acccov_decoder(x))
        gyro = self.gyro_cov_min + (self.gyro_cov_max - self.gyro_cov_min) * torch.sigmoid(self.gyrocov_decoder(x))

        return torch.cat([acc, gyro], dim = -1)

    def decoder(self, x):
        acc = self.accdecoder(x) * self.acc_std
        gyro = self.gyrodecoder(x) * self.gyro_std

        return torch.cat([acc, gyro], dim = -1)

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
            cov = self.cov_decoder(feature)
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

        # Only the raw integration below supplies covariance to the loss.
        out_state = self.integrate(
            init_state=init_state, data=data,
            cov_state=inference_state['cov_state'], propagate_cov=False,
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

        if self.conf.propcov:
            raw_data = dict(data)
            raw_data['corrected_acc'] = data['acc'][:, self.interval:, :]
            raw_data['corrected_gyro'] = data['gyro'][:, self.interval:, :]
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
            cov = self.cov_decoder(feature)
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
