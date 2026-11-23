"""Parameter-space objective diagnostics without changing optimizer gradients."""

import math

import torch


def gradient_norm(values):
    terms = [value.detach().double().square().sum() for value in values if value is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.


def clipping_groups(optimizer_groups, *, mode="global", global_limit=1., gru2_limit=5., head_limit=1.,
                    cnn_limit=1., gru1_limit=1.):
    """Define disjoint clipping sets, independent of Adam's learning-rate groups."""
    if mode not in ("global", "separate"):
        raise ValueError("Clipping mode must be global or separate.")
    limits = (global_limit,) if mode == "global" else (gru2_limit, head_limit, cnn_limit, gru1_limit)
    if any(not math.isfinite(value) or value <= 0 for value in limits):
        raise ValueError("Gradient clipping limits must be finite and positive.")
    names = [group.get("name", f"group_{i}") for i, group in enumerate(optimizer_groups)]
    parameters = [p for group in optimizer_groups for p in group["params"]]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Clipping sets must not contain duplicate parameters.")
    if mode == "global":
        return [{"name": "global", "optimizer_groups": names,
                 "params": parameters, "max_norm": global_limit}]
    allowed = {"cnn", "gru1", "gru2", "covariance_heads", "bias_gains", "raw_cov_adapter"}
    if "gru2" not in names or set(names)-allowed:
        raise ValueError("Separate clipping requires named encoder/head/gain groups and GRU2.")
    result = []
    for name, selected, limit in (("cnn", {"cnn"}, cnn_limit), ("gru1", {"gru1"}, gru1_limit),
                                   ("gru2", {"gru2"}, gru2_limit),
                                   ("heads_and_bias_gains", {"covariance_heads", "bias_gains", "raw_cov_adapter"}, head_limit)):
        matched = [group for group in optimizer_groups if group["name"] in selected]
        params = [p for group in matched for p in group["params"]]
        if not params:
            if name in ("cnn", "gru1"):
                continue
            raise ValueError(f"Empty clipping set: {name}")
        result.append({"name": name, "optimizer_groups": [group["name"] for group in matched],
                       "params": params, "max_norm": limit})
    return result


def clipping_policy(groups, mode):
    return {"mode": mode, "groups": {
        group["name"]: {key: group[key] for key in ("optimizer_groups", "max_norm")}
        for group in groups}}


def clip_gradients(groups):
    """Apply exactly one clip to each parameter; never apply a second global clip."""
    result = {}
    for group in groups:
        before = float(torch.nn.utils.clip_grad_norm_(
            group["params"], group["max_norm"], error_if_nonfinite=True))
        result[group["name"]] = {
            "max_norm": group["max_norm"], "norm_before": before,
            "norm_after": gradient_norm(p.grad for p in group["params"]),
            "was_clipped": before > group["max_norm"],
            "scale": min(1., group["max_norm"]/(before+1e-6))}
    return result


def parameter_update_diagnostics(optimizer_groups, before, max_relative_update=None):
    """Measure actual Adam changes, including moment normalization, by LR group."""
    result = {}
    for index, group in enumerate(optimizer_groups):
        name = group.get("name", f"group_{index}")
        originals = [before[id(p)] for p in group["params"]]
        deltas = [p.detach()-old for p, old in zip(group["params"], originals)]
        norm = gradient_norm(deltas)
        parameter_norm = gradient_norm(originals)
        # Bias gains can start at exactly zero. A unit L2 floor avoids treating
        # their first ordinary Adam step as an infinitely large relative step.
        normalization = max(parameter_norm, 1.)
        relative = norm/normalization
        maximum = max(float(delta.abs().max()) for delta in deltas)
        if not all(math.isfinite(value) for value in (norm, parameter_norm, relative, maximum)):
            raise RuntimeError(f"Optimization instability: non-finite Adam update in {name}.")
        if max_relative_update is not None and relative > max_relative_update:
            raise RuntimeError(f"Optimization instability: {name} relative Adam update "
                               f"{relative:.6g} exceeds {max_relative_update:.6g} "
                               f"(update norm={norm:.6g}, parameter norm={parameter_norm:.6g}).")
        result[name] = {"l2": norm, "rms": norm/math.sqrt(sum(p.numel() for p in group["params"])),
                        "max_abs": maximum, "parameter_norm_before": parameter_norm,
                        "normalization_norm": normalization,
                        "relative_l2": relative, "learning_rate": group["lr"]}
    return result


def component_gradient_diagnostics(variances, outputs, parameters, groups, gyro_weight):
    """Measure each batch-mean VJP; leave .grad unset for the actual update.

    Gyro's weighted norm is its contribution to the optimizer objective. Its
    unweighted norm is also logged so lambda_g cannot hide the raw loss scale.
    These are parameter gradients, not gradients in variance-output space.
    """
    components = {}
    for name, key in (("velocity", "velocity_gradient"),
                      ("gyro_calibration_unweighted", "gyro_calibration_gradient")):
        values = [torch.as_tensor(row[key], device=q.device, dtype=q.dtype)/len(outputs)
                  for q, row in zip(variances, outputs)]
        components[name] = torch.autograd.grad(tuple(variances), tuple(parameters),
            grad_outputs=values, retain_graph=True, allow_unused=True)
    velocity = components["velocity"]
    gyro = components["gyro_calibration_unweighted"]
    velocity_norm = gradient_norm(velocity)
    gyro_raw_norm = gradient_norm(gyro)
    if not all(math.isfinite(x) for x in (velocity_norm,gyro_raw_norm)):
        raise RuntimeError('Non-finite component parameter gradient.')
    gyro_norm = abs(gyro_weight)*gyro_raw_norm
    dot_terms = [(v.detach().double()*g.detach().double()).sum()
                 for v,g in zip(velocity,gyro) if v is not None and g is not None]
    dot = float(torch.stack(dot_terms).sum())*gyro_weight if dot_terms else 0.
    cosine = max(-1., min(1., dot/(velocity_norm*gyro_norm))) if velocity_norm and gyro_norm else None
    by_group = {}
    for index,group in enumerate(groups):
        ids = {id(p) for p in group['params']}
        selected = [i for i,p in enumerate(parameters) if id(p) in ids]
        v = gradient_norm(velocity[i] for i in selected)
        g = gradient_norm(gyro[i] for i in selected)
        by_group[group.get('name',f'group_{index}')] = {'velocity':v, 'gyro_calibration':abs(gyro_weight)*g,
                                  'gyro_calibration_unweighted':g}
    return {'gradient_norm_velocity':velocity_norm,
            'gradient_norm_gyro_calibration':gyro_norm,
            'gradient_norm_gyro_calibration_unweighted':gyro_raw_norm,
            'gradient_gyro_to_velocity_ratio':gyro_norm/velocity_norm if velocity_norm else None,
            'gradient_velocity_gyro_cosine':cosine,
            'gradient_norms_by_group':by_group}


def summarize_batches(rows):
    """One observation per optimizer step, never duplicated per-flight norms."""
    if not rows:
        return {}
    def stats(values):
        ordered = sorted(values)
        def quantile(p):
            x = (len(ordered)-1)*p
            i = int(x)
            return ordered[i]+(ordered[min(i+1,len(ordered)-1)]-ordered[i])*(x-i)
        return {'min':ordered[0], 'mean':sum(ordered)/len(ordered),
                'median':quantile(.5), 'p95':quantile(.95), 'max':ordered[-1]}
    keys = ('gradient_norm_before_clip', 'gradient_norm_after_clip',
            'gradient_norm_adapter_regularization',
            'gradient_norm_velocity', 'gradient_norm_gyro_calibration',
            'gradient_norm_gyro_calibration_unweighted', 'gradient_gyro_to_velocity_ratio',
            'gradient_velocity_gyro_cosine', 'gradient_clip_scale')
    summary = {'batches':len(rows), 'clipped_batches':sum(row['gradient_was_clipped'] for row in rows)}
    summary['clip_fraction'] = summary['clipped_batches']/len(rows)
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            if not all(math.isfinite(x) for x in values):
                raise RuntimeError(f'Non-finite gradient diagnostic: {key}')
            summary[key] = stats(values)
    summary['by_group'] = {}
    for name in sorted({name for row in rows for name in row.get('gradient_norms_by_group',{})}):
        group = {}
        for component in ('velocity','gyro_calibration','gyro_calibration_unweighted','total','total_after_clip'):
            values = [row['gradient_norms_by_group'][name][component] for row in rows
                      if component in row.get('gradient_norms_by_group',{}).get(name,{})]
            if values:
                group[component] = stats(values)
        summary['by_group'][name] = group
    for source, target, fields in (
        ('gradient_clipping_groups', 'clipping_groups', ('norm_before','norm_after','scale')),
        ('parameter_updates_by_group', 'parameter_updates_by_group', ('l2','rms','max_abs','relative_l2')),
    ):
        names = sorted({name for row in rows for name in row.get(source,{})})
        if names:
            summary[target] = {}
        for name in names:
            values = [row[source][name] for row in rows if name in row.get(source,{})]
            entry = {field:stats([value[field] for value in values]) for field in fields}
            if source == 'gradient_clipping_groups':
                entry.update(max_norm=values[0]['max_norm'],
                             clip_fraction=sum(value['was_clipped'] for value in values)/len(values))
            summary[target][name] = entry
    return summary
