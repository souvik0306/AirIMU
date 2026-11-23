"""Strict checkpoint validation, with an explicit Stage 1 -> adapter migration."""

import torch


def covariance_policy(network):
    return {name: bool(getattr(network, name)) if name == "use_cov_limits" else float(getattr(network, name))
            for name in ("use_cov_limits", "acc_cov_min", "gyro_cov_min", "acc_cov_max", "gyro_cov_max",
                         "acc_cov_target", "gyro_cov_target")}


def validate_covariance_policy(network, saved):
    if saved is None:
        return  # Legacy checkpoints require a matching user-supplied config.
    current = covariance_policy(network)
    for key, value in current.items():
        if key.endswith("_max") and not current["use_cov_limits"]:
            continue
        if value != saved.get(key):
            raise ValueError(f"Checkpoint covariance policy differs in {key}: {saved.get(key)} != {value}")


def load_covariance_checkpoint(network, state, *, initialize_adapter=False):
    expected = network.state_dict()
    allowed_missing = {key for key in expected if key.startswith("raw_cov_adapter.")} if initialize_adapter else set()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if unexpected or missing != allowed_missing:
        raise ValueError(f"Checkpoint keys differ: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name, value in state.items():
        if not torch.is_tensor(value) or value.shape != expected[name].shape:
            raise ValueError(f"Invalid checkpoint tensor shape: {name}")
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
    # strict=True after supplying only the explicitly new, zero-output adapter.
    network.load_state_dict({**expected, **state}, strict=True)


def configure_stage2_parameters(network, conf):
    """Freeze Stage 1; optional covariance-head tuning uses its own low LR."""
    if not conf.get("raw_cov_adapter", False):
        return [{"params": list(network.parameters()), "lr": conf.lr}]
    if network.raw_cov_adapter is None:
        raise ValueError("Stage 2 adapter is missing.")
    network.requires_grad_(False)
    network.raw_cov_adapter.requires_grad_(True)
    groups = [{"name": "raw_cov_adapter", "params": list(network.raw_cov_adapter.parameters()), "lr": conf.lr}]
    if conf.get("adapter_finetune_heads", False):
        lr = float(conf.get("adapter_head_lr", conf.lr*.1))
        if not 0 < lr < conf.lr:
            raise ValueError("Adapter fine-tuning needs 0 < adapter_head_lr < lr.")
        parameters = []
        for module in (network.acccov_decoder, network.gyrocov_decoder):
            module.requires_grad_(True)
            parameters.extend(module.parameters())
        groups.append({"name": "stage1_covariance_heads", "params": parameters, "lr": lr})
    return groups
