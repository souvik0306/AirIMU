import torch

def imu_seq_collate(data):
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    dt = torch.stack([d['dt'] for d in data])

    return {
        'dt': dt,
        'acc': acc,
        'gyro': gyro,

        'gt_pos': gt_pos,
        'gt_vel': gt_vel,
        'gt_rot': gt_rot,

        'init_pos': init_pos,
        'init_vel': init_vel,
        'init_rot': init_rot,
    }

def custom_collate(data):
    dt = torch.stack([d['dt'] for d in data])
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])
    rot = torch.stack([d['rot'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    return  {'dt': dt, 'acc': acc, 'gyro': gyro, 'rot':rot,}, \
            {'pos': init_pos, 'vel': init_vel, 'rot': init_rot,}, \
            {'gt_pos': gt_pos, 'gt_vel': gt_vel, 'gt_rot': gt_rot, }

def padding_collate(data, pad_len = 1, use_gravity = True):
    B = len(data)
    input_data, init_state, label = custom_collate(data)

    original_length = input_data['acc'].shape[1]
    
    # Print detailed padding info (only for first call)
    if not hasattr(padding_collate, '_first_call_done'):
        print("\n" + "="*80)
        print("PADDING COLLATE - DETAILED BREAKDOWN")
        print("="*80)
        
        print(f"\n1. INPUT FROM DATASET (before padding):")
        print(f"   - Batch size: {B}")
        print(f"   - Window length: {original_length}")
        print(f"   - acc shape: {input_data['acc'].shape}")
        print(f"   - gyro shape: {input_data['gyro'].shape}")
        print(f"   - First 3 acc samples from dataset:")
        for i in range(min(3, original_length)):
            print(f"     acc[{i}] = {input_data['acc'][0, i, :].tolist()}")
        print(f"   - First 3 gyro samples from dataset:")
        for i in range(min(3, original_length)):
            print(f"     gyro[{i}] = {input_data['gyro'][0, i, :].tolist()}")
        
        print(f"\n2. INITIAL STATE (used for padding generation):")
        print(f"   - init_rot shape: {init_state['rot'].shape}")
        print(f"   - init_rot value (quaternion [qx,qy,qz,qw]): {init_state['rot'][0, 0, :].tolist()}")
        print(f"   - init_pos: {init_state['pos'][0, 0, :].tolist()}")
        print(f"   - init_vel: {init_state['vel'][0, 0, :].tolist()}")
        
        padding_collate._first_call_done = True
    
    if use_gravity:
        iden_acc_vector = torch.tensor([0.,0.,9.81007], dtype=input_data['dt'].dtype).repeat(B,pad_len,1)
    else:
        iden_acc_vector = torch.zeros(B, pad_len, 3, dtype=input_data['dt'].dtype)

    pad_acc = init_state['rot'].Inv() * iden_acc_vector
    pad_gyro = torch.zeros(B, pad_len, 3, dtype=input_data['dt'].dtype)
    
    # Print padding details (only for first call)
    if not hasattr(padding_collate, '_padding_printed'):
        print(f"\n3. SYNTHETIC PADDING GENERATED:")
        print(f"   - Padding length: {pad_len} frames")
        print(f"   - Gravity vector (world frame): [0, 0, 9.81007]")
        print(f"   - Padding acc (gravity rotated to body frame):")
        for i in range(pad_len):
            print(f"     pad_acc[{i}] = {pad_acc[0, i, :].tolist()}")
        print(f"   - Padding gyro (all zeros - stationary):")
        for i in range(min(3, pad_len)):
            print(f"     pad_gyro[{i}] = {pad_gyro[0, i, :].tolist()}")
        padding_collate._padding_printed = True

    input_data["acc"] = torch.cat([pad_acc, input_data['acc']], dim =1)
    input_data["gyro"] = torch.cat([pad_gyro, input_data['gyro']], dim =1)
    
    # Print final concatenated result (only for first call)
    if not hasattr(padding_collate, '_concat_printed'):
        print(f"\n4. AFTER CONCATENATION (padding + real data):")
        print(f"   - Final acc shape: {input_data['acc'].shape}")
        print(f"   - Final gyro shape: {input_data['gyro'].shape}")
        print(f"   - Structure: [{pad_len} padding frames] + [{original_length} real frames]")
        print(f"   - First {pad_len+3} acc samples (padding + first 3 real):")
        for i in range(min(pad_len+3, input_data['acc'].shape[1])):
            marker = "(PAD)" if i < pad_len else "(REAL)"
            print(f"     acc[{i}] = {input_data['acc'][0, i, :].tolist()} {marker}")
        print("="*80 + "\n")
        padding_collate._concat_printed = True
    
    return  input_data, init_state, label

collate_fcs ={
    "base": custom_collate,
    "padding": padding_collate,
    "padding9": lambda data: padding_collate(data, pad_len = 9),
    "padding1": lambda data: padding_collate(data, pad_len = 1),
    "Gpadding": lambda data: padding_collate(data, pad_len = 9, use_gravity = False),
}