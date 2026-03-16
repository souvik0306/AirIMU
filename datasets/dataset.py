import argparse
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.utils.data as Data
from pyhocon import ConfigFactory


class Sequence(ABC):
    # Dictionary to keep track of subclasses
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

class SeqDataset(Data.Dataset):
    def __init__(self, root, dataname, device = 'cpu', name='Nav', duration=200, step_size=200, mode='inference', 
                    drop_last = True, conf = {}):
        super().__init__()

        self.DataClass = Sequence.subclasses
        
        self.conf = conf
        self.seq = self.DataClass[name](root, dataname, **self.conf)
        self.data = self.seq.data
        self.seqlen = self.seq.get_length()-1
        self.gravity = conf.gravity if "gravity" in conf.keys() else 9.81007
        if duration is None: self.duration = self.seqlen
        else: self.duration = duration
        
        if step_size is None: self.step_size = self.seqlen
        else: self.step_size = step_size

        self.data['acc_cov'] = 0.08 * torch.ones_like(self.data['acc'])
        self.data['gyro_cov'] = 0.006 * torch.ones_like(self.data['gyro'])

        start_frame = 0
        end_frame = self.seqlen

        self.index_map = [[i, i + self.duration] for i in range(
            0, end_frame - start_frame - self.duration, self.step_size)]
        if (self.index_map[-1][-1] < end_frame) and (not drop_last):
            self.index_map.append([self.index_map[-1][-1], end_frame])

        self.index_map = np.array(self.index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id, end_frame_id = self.index_map[i]
        return {
            'dt': self.data['dt'][frame_id: end_frame_id],
            'acc': self.data['acc'][frame_id: end_frame_id],
            'gyro': self.data['gyro'][frame_id: end_frame_id],
            'rot': self.data['gt_orientation'][frame_id: end_frame_id],
            'gt_pos': self.data['gt_translation'][frame_id+1: end_frame_id+1],
            'gt_rot': self.data['gt_orientation'][frame_id+1: end_frame_id+1],
            'gt_vel': self.data['velocity'][frame_id+1: end_frame_id+1],
            'init_pos': self.data['gt_translation'][frame_id][None, ...],
            'init_rot': self.data['gt_orientation'][frame_id: end_frame_id],
            'init_vel': self.data['velocity'][frame_id][None, ...],
        }

    def get_init_value(self):
        return {'pos': self.data['gt_translation'][:1],
                'rot': self.data['gt_orientation'][:1],
                'vel': self.data['velocity'][:1]}

    def get_mask(self):
        return self.data['mask']
    
    def get_gravity(self):
        return self.gravity


class SeqInfDataset(SeqDataset):
    @staticmethod
    def _as_time_series(tensor):
        # Accept [T, C] or [1, T, C] and return [T, C].
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            return tensor[0]
        return tensor

    @staticmethod
    def _align_series_length(tensor, target_len, name):
        cur_len = tensor.shape[0]
        if cur_len == target_len:
            return tensor
        if cur_len > target_len:
            print(f"[SeqInfDataset] Trimming {name} from {cur_len} to {target_len}")
            return tensor[:target_len]
        if cur_len <= 0:
            raise ValueError(f"{name} is empty and cannot be aligned to target length {target_len}")

        repeat_factor = int(np.ceil(target_len / cur_len))
        print(f"[SeqInfDataset] Expanding {name} from {cur_len} to {target_len} using repeat factor {repeat_factor}")
        return tensor.repeat_interleave(repeat_factor, dim=0)[:target_len]

    def __init__(self, root, dataname, inference_state, device =  'cpu', name='Nav', duration=200, step_size=200, 
                            drop_last = True, mode='inference', usecov = True, useraw = False,conf={}):
        super().__init__(root, dataname, device, name, duration, step_size, mode, drop_last, conf)
        target_imu_len = self.data['acc'][:-1].shape[0]

        correction_acc = self._as_time_series(inference_state['correction_acc'].cpu())
        correction_gyro = self._as_time_series(inference_state['correction_gyro'].cpu())

        correction_acc = self._align_series_length(correction_acc, target_imu_len, 'correction_acc')
        correction_gyro = self._align_series_length(correction_gyro, target_imu_len, 'correction_gyro')

        self.data['acc'][:-1] += correction_acc.to(self.data['acc'].dtype)
        self.data['gyro'][:-1] += correction_gyro.to(self.data['gyro'].dtype)
       
        if 'acc_cov' in inference_state.keys() and usecov:
            target_full_len = self.data['acc'].shape[0]
            acc_cov = self._as_time_series(inference_state['acc_cov'].cpu())
            acc_cov = self._align_series_length(acc_cov, target_full_len, 'acc_cov')
            self.data['acc_cov'] = acc_cov.to(self.data['acc'].dtype)

        if 'gyro_cov' in inference_state.keys() and usecov:
            target_full_len = self.data['gyro'].shape[0]
            gyro_cov = self._as_time_series(inference_state['gyro_cov'].cpu())
            gyro_cov = self._align_series_length(gyro_cov, target_full_len, 'gyro_cov')
            self.data['gyro_cov'] = gyro_cov.to(self.data['gyro'].dtype)


class SeqeuncesDataset(Data.Dataset):
    """
    For the purpose of training and inferering
    1. Abandon the features of the last time frame, since there are no ground truth pose and dt
     to integrate the imu data of the last frame. So the length of the dataset is seq.get_length() - 1
    """
    def __init__(self, data_set_config, mode = None, data_path = None, data_root = None, device= "cuda:0", n_freq=1, n_mode="interval"):
        super(SeqeuncesDataset, self).__init__()
        (
            self.ts,
            self.dt,
            self.acc,
            self.gyro,
            self.gt_pos,
            self.gt_ori,
            self.gt_velo,
            self.index_map,
            self.seq_idx,
        ) = ([], [], [], [], [], [], [], [], 0)
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))
        self.device = device
        self.conf = data_set_config
        self.gravity = data_set_config.gravity if "gravity" in data_set_config.keys() else 9.81007
        self.n_freq = max(1, int(n_freq))
        self.n_mode = (n_mode or "interval").lower()
        if mode is None:
            self.mode = data_set_config.mode
        else:
            self.mode = mode

        self.DataClass = Sequence.subclasses

        ## the design of datapath provide a quick way to revisit a specific sequence, but introduce some inconsistency
        if data_path is None:
            for conf in data_set_config.data_list:
                for path in conf.data_drive:
                    self.construct_index_map(conf, conf["data_root"], path, self.seq_idx)
                    self.seq_idx += 1
        ## the design of dataroot provide a quick way to introduce multiple sequences in eval set, but introduce some inconsistency
        elif data_root is None:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, conf["data_root"], data_path, self.seq_idx)
            self.seq_idx += 1
        else:
            conf = data_set_config.data_list[0]
            self.construct_index_map(conf, data_root, data_path, self.seq_idx)
            self.seq_idx += 1

    def _block_mean(self, tensor, block_size):
        chunks = [
            tensor[i : i + block_size].mean(dim=0, keepdim=True)
            for i in range(0, tensor.shape[0], block_size)
        ]
        return torch.cat(chunks, dim=0)

    def _use_frequency_mode(self):
        return self.n_freq > 1 and self.mode in {"inference", "infevaluate", "evaluate"}

    def _effective_length(self, length):
        if not self._use_frequency_mode():
            return int(length)
        return int(np.ceil(length / self.n_freq))

    def _quat_block_mean(self, quat, block_size):
        quat_mean = self._block_mean(quat, block_size)
        quat_norm = torch.linalg.norm(quat_mean, dim=-1, keepdim=True).clamp_min(1e-12)
        return quat_mean / quat_norm

    def _sync_lengths(self, dt, acc, gyro, gt_pos, gt_ori, gt_velo, ref_dt_last, ref_gt_pos_last, ref_gt_ori_last, ref_gt_velo_last):
        imu_target = acc.shape[0]
        gt_target = imu_target + 1
        dt_target = imu_target + 1

        if gt_pos.shape[0] < gt_target:
            pad_count = gt_target - gt_pos.shape[0]
            gt_pos = torch.cat([gt_pos, ref_gt_pos_last.repeat(pad_count, *([1] * (ref_gt_pos_last.dim() - 1)))], dim=0)
            gt_ori = torch.cat([gt_ori, ref_gt_ori_last.repeat(pad_count, *([1] * (ref_gt_ori_last.dim() - 1)))], dim=0)
            gt_velo = torch.cat([gt_velo, ref_gt_velo_last.repeat(pad_count, *([1] * (ref_gt_velo_last.dim() - 1)))], dim=0)
        else:
            gt_pos = gt_pos[:gt_target]
            gt_ori = gt_ori[:gt_target]
            gt_velo = gt_velo[:gt_target]

        if dt.shape[0] < dt_target:
            pad_count = dt_target - dt.shape[0]
            dt = torch.cat([dt, ref_dt_last.repeat(pad_count, *([1] * (ref_dt_last.dim() - 1)))], dim=0)
        else:
            dt = dt[:dt_target]

        return dt, acc, gyro, gt_pos, gt_ori, gt_velo

    def _apply_frequency_mode(self, dt, acc, gyro, gt_pos, gt_ori, gt_velo):
        if not self._use_frequency_mode():
            return dt, acc, gyro, gt_pos, gt_ori, gt_velo

        mode = self.n_mode
        if mode not in {"interval", "avg", "average"}:
            raise ValueError(f"Unsupported n_mode={self.n_mode}. Use 'interval' or 'avg'.")

        ref_dt_last = dt[-1:]
        ref_gt_pos_last = gt_pos[-1:]
        ref_gt_ori_last = gt_ori[-1:]
        ref_gt_velo_last = gt_velo[-1:]

        if mode == "interval":
            imu_idx = torch.arange(0, acc.shape[0], self.n_freq, device=acc.device)
            state_idx = torch.cat([
                imu_idx,
                torch.tensor([acc.shape[0]], device=acc.device, dtype=imu_idx.dtype),
            ])

            acc = acc.index_select(0, imu_idx)
            gyro = gyro.index_select(0, imu_idx)
            gt_pos = gt_pos.index_select(0, state_idx.clamp_max(gt_pos.shape[0] - 1))
            gt_ori = gt_ori.index_select(0, state_idx.clamp_max(gt_ori.shape[0] - 1))
            gt_velo = gt_velo.index_select(0, state_idx.clamp_max(gt_velo.shape[0] - 1))

            dt_steps = []
            for i in range(state_idx.shape[0] - 1):
                s = int(state_idx[i].item())
                e = int(state_idx[i + 1].item())
                dt_steps.append(dt[s:e].sum(dim=0, keepdim=True))
            dt = torch.cat(dt_steps + [dt_steps[-1].clone()], dim=0)
        else:
            acc = self._block_mean(acc, self.n_freq)
            gyro = self._block_mean(gyro, self.n_freq)
            gt_pos = self._block_mean(gt_pos, self.n_freq)
            gt_ori = self._quat_block_mean(gt_ori, self.n_freq)
            gt_velo = self._block_mean(gt_velo, self.n_freq)

            dt_steps = self._block_mean(dt[:-1], self.n_freq)
            dt = torch.cat([dt_steps, dt_steps[-1:].clone()], dim=0)

        return self._sync_lengths(
            dt,
            acc,
            gyro,
            gt_pos,
            gt_ori,
            gt_velo,
            ref_dt_last,
            ref_gt_pos_last,
            ref_gt_ori_last,
            ref_gt_velo_last,
        )

    def load_data(self, seq, start_frame, end_frame):
        dt = seq.data["dt"][start_frame:end_frame+1]
        acc = seq.data["acc"][start_frame:end_frame]
        gyro = seq.data["gyro"][start_frame:end_frame]
        gt_pos = seq.data["gt_translation"][start_frame:end_frame+1]
        gt_ori = seq.data["gt_orientation"][start_frame:end_frame+1]
        gt_velo = seq.data["velocity"][start_frame:end_frame+1]

        dt, acc, gyro, gt_pos, gt_ori, gt_velo = self._apply_frequency_mode(
            dt, acc, gyro, gt_pos, gt_ori, gt_velo
        )

        if "time" in seq.data.keys():
            ts = seq.data["time"][start_frame:end_frame]
            if self.n_freq > 1:
                if self.n_mode == "interval":
                    ts = ts[:: self.n_freq]
                elif self.n_mode in {"avg", "average"}:
                    ts = self._block_mean(ts, self.n_freq)
            self.ts.append(ts)

        self.acc.append(acc)
        self.gyro.append(gyro)
        # the groud truth state should include the init state and integrated state, thus has one more frame than imu data
        self.dt.append(dt)
        self.gt_pos.append(gt_pos)
        self.gt_ori.append(gt_ori)
        self.gt_velo.append(gt_velo)

    def construct_index_map(self, conf, data_root, data_name, seq_id):
        seq = self.DataClass[conf.name](data_root, data_name, intepolate = True, **self.conf)
        seq_len = seq.get_length() -1 # abandon the last imu features
        window_size, step_size = conf.window_size, conf.step_size
        ## seting the starting and ending duration with different trianing mode
        start_frame, end_frame = 0, seq_len

        if self.mode == 'train_half':
            end_frame = np.floor(seq_len * 0.5).astype(int)
        elif self.mode == 'test_half':
            start_frame = np.floor(seq_len * 0.5).astype(int)
        elif self.mode == 'train_1m':
            end_frame = 12000
        elif self.mode == 'test_1m':
            start_frame = 12000
        elif self.mode == 'mini':# For the purpse of debug
            end_frame = 1000

        _duration = end_frame - start_frame
        effective_duration = self._effective_length(_duration)
        effective_window = max(1, self._effective_length(window_size))
        effective_step = max(1, self._effective_length(step_size))

        if self.mode == "inference":
            self.index_map = [[seq_id, 0, effective_duration]]
        elif self.mode == "infevaluate":
            windows = [
                [seq_id, j, j+effective_window] for j in range(
                    0, effective_duration - effective_window, effective_step)
            ]
            if not windows:
                windows = [[seq_id, 0, effective_duration]]
            self.index_map += windows
            if self.index_map and self.index_map[-1][2] < effective_duration:
                print(self.index_map[-1][2])
                self.index_map += [[seq_id, self.index_map[-1][2], effective_duration]]
        elif self.mode == 'evaluate':
            # adding the last piece for evaluation
            windows = [
                [seq_id, j, j+effective_window] for j in range(
                    0, effective_duration - effective_window, effective_step)
            ]
            if not windows:
                windows = [[seq_id, 0, effective_duration]]
            self.index_map += windows
        elif self.mode == 'train_half_random':
            np.random.seed(1)   
            window_group_size = 3000
            selected_indices = [j for j in range(0, _duration-window_group_size, window_group_size)]
            np.random.shuffle(selected_indices)
            indices_num = len(selected_indices)
            for w in selected_indices[:np.floor(indices_num * 0.5).astype(int)]:  
                self.index_map +=[[seq_id, j, j + window_size] for j in range(w, w+window_group_size-window_size,step_size)]
        elif self.mode == 'test_half_random':
            np.random.seed(1)
            window_group_size = 3000
            selected_indices = [j for j in range(0, _duration-window_group_size, window_group_size)]
            np.random.shuffle(selected_indices)
            indices_num = len(selected_indices)
            for w in selected_indices[np.floor(indices_num * 0.5).astype(int):]:   
                self.index_map +=[[seq_id, j, j + window_size] for j in range(w, w+window_group_size-window_size,step_size)]  
        else:
            ## applied the mask if we need the training.
            self.index_map +=[
                [seq_id, j, j+window_size] for j in range(
                    0, _duration - window_size, step_size)
                    if torch.all(seq.data["mask"][j: j+window_size])
            ]
        
        ## Loading the data from each sequence into 
        self.load_data(seq, start_frame, end_frame)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id, end_frame_id = self.index_map[item][0], self.index_map[item][1], self.index_map[item][2]
        data = {
            'dt': self.dt[seq_id][frame_id: end_frame_id],
            'acc': self.acc[seq_id][frame_id: end_frame_id],
            'gyro': self.gyro[seq_id][frame_id: end_frame_id],
            'rot': self.gt_ori[seq_id][frame_id: end_frame_id]
        }
        init_state = {
            'init_rot': self.gt_ori[seq_id][frame_id][None, ...],
            'init_pos': self.gt_pos[seq_id][frame_id][None, ...],
            'init_vel': self.gt_velo[seq_id][frame_id][None, ...],
        }
        label = {
            'gt_pos': self.gt_pos[seq_id][frame_id+1 : end_frame_id+1],
            'gt_rot': self.gt_ori[seq_id][frame_id+1 : end_frame_id+1],
            'gt_vel': self.gt_velo[seq_id][frame_id+1 : end_frame_id+1],
        }

        return {**data, **init_state, **label}

    def get_dtype(self):
        return self.acc[0].dtype
    


if __name__ == '__main__':
    from datasets.dataset_utils import custom_collate
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/datasets/BaselineEuRoC.conf', help='config file path, i.e., configs/Euroc.conf')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")

    args = parser.parse_args(); print(args)
    conf = ConfigFactory.parse_file(args.config)
    
    dataset = SeqeuncesDataset(data_set_config=conf.train)
    loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    for i, (data, init, label) in enumerate(loader):
        for k in data: print(k, ":", data[k].shape)
        for k in init: print(k, ":", init[k].shape)
        for k in label: print(k, ":", label[k].shape)
