import os
import torch
import numpy as np
import pypose as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _safe_drive_name(save_prefix):
    return str(save_prefix).replace(os.sep, "_").replace(" ", "_")


def _category_dir(save_folder, save_prefix, category):
    drive_name = _safe_drive_name(save_prefix)
    category_path = os.path.join(save_folder, drive_name, category)
    os.makedirs(category_path, exist_ok=True)
    return category_path


def visualize_state_error(save_prefix, relative_outstate, relative_infstate, \
                            save_folder=None, mask=None, file_name="state_error_compare.png"):
    if mask is None:
        outstate_pos_err = relative_outstate['pos_dist'][0]
        outstate_vel_err = relative_outstate['vel_dist'][0]
        outstate_rot_err = relative_outstate['rot_dist'][0]
        
        infstate_pos_err = relative_infstate['pos_dist'][0]
        infstate_vel_err = relative_infstate['vel_dist'][0]
        infstate_rot_err = relative_infstate['rot_dist'][0]
    else:
        outstate_pos_err = relative_outstate['pos_dist'][0, mask]
        outstate_vel_err = relative_outstate['vel_dist'][0, mask]
        outstate_rot_err = relative_outstate['rot_dist'][0, mask]
        
        infstate_pos_err = relative_infstate['pos_dist'][0, mask]
        infstate_vel_err = relative_infstate['vel_dist'][0, mask]
        infstate_rot_err = relative_infstate['rot_dist'][0, mask]
    
    fig, axs = plt.subplots(3,)
    fig.suptitle("Integration error vs AirIMU Integration error")
    
    axs[0].plot(outstate_pos_err,color = 'b',linewidth=1)
    axs[0].plot(infstate_pos_err,color = 'red',linewidth=1)
    axs[0].legend(["integration_pos_error", "AirIMU_pos_error"])
    axs[0].grid(True)
    
    axs[1].plot(outstate_vel_err,color = 'b',linewidth=1)
    axs[1].plot(infstate_vel_err,color = 'red',linewidth=1)
    axs[1].legend(["integration_vel_error", "AirIMU_vel_error"])
    axs[1].grid(True)
    
    axs[2].plot(outstate_rot_err,color = 'b',linewidth=1)
    axs[2].plot(infstate_rot_err,color = 'red',linewidth=1)
    axs[2].legend(["integration_rot_error", "AirIMU_rot_error"])
    axs[2].grid(True)
    
    plt.tight_layout()
    if save_folder is not None:
        category_path = _category_dir(save_folder, save_prefix, "position")
        full_path = os.path.join(category_path, file_name)
        plt.savefig(full_path, dpi = 600)
    plt.close(fig)
  

def visualize_rotations(save_prefix, gt_rot, out_rot, inf_rot = None,save_folder=None):
   
    gt_euler = 180./np.pi* pp.SO3(gt_rot).euler()
    outstate_euler = 180./np.pi* pp.SO3(out_rot).euler()
    
    legend_list = ["roll","pitch", "yaw"]
    fig, axs = plt.subplots(3,)
    fig.suptitle("integrated orientation")
    for i in range(3):
        axs[i].plot(outstate_euler[:,i],color = 'b',linewidth=0.9)
        axs[i].plot(gt_euler[:,i],color = 'mediumseagreen',linewidth=0.9)
        axs[i].legend(["Integrated_"+legend_list[i],"gt_"+legend_list[i]])
        axs[i].grid(True)
    
    if inf_rot is not None:
        infstate_euler = 180./np.pi* pp.SO3(inf_rot).euler()
        print(infstate_euler.shape)
        for i in range(3):
            axs[i].plot(infstate_euler[:,i],color = 'red',linewidth=0.9)
            axs[i].legend(["Integrated_"+legend_list[i],"gt_"+legend_list[i],"AirIMU_"+legend_list[i]])
    plt.tight_layout()
    if save_folder is not None:
        category_path = _category_dir(save_folder, save_prefix, "orientation")
        full_path = os.path.join(category_path, "orientation_compare.png")
        plt.savefig(full_path, dpi = 600)
    plt.close(fig)


def visualize_trajectory(save_prefix, save_folder, outstate, infstate):
    gt_x, gt_y, gt_z                = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    rawTraj_x, rawTraj_y, rawTraj_z = torch.split(outstate["poses"][0].cpu(), 1, dim=1)
    airTraj_x, airTraj_y, airTraj_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)
    
    fig, ax = plt.subplots()
    ax.plot(rawTraj_x, rawTraj_y, label="Raw")
    ax.plot(airTraj_x, airTraj_y, label="AirIMU")
    ax.plot(gt_x     , gt_y     , label="Ground Truth")
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    category_path = _category_dir(save_folder, save_prefix, "position")
    full_path = os.path.join(category_path, "trajectory_xy.png")
    plt.savefig(full_path, dpi = 600)
    plt.close()
    
    ###########################################################
    
    fig, ax = plt.subplots()
    ax.plot(rawTraj_x, rawTraj_z, label="Raw")
    ax.plot(airTraj_x, airTraj_z, label="AirIMU")
    ax.plot(gt_x     , gt_z     , label="Ground Truth")
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    full_path = os.path.join(category_path, "trajectory_xz.png")
    plt.savefig(full_path, dpi = 600)
    plt.close()
    
    ###########################################################
    
    fig, ax = plt.subplots()
    ax.plot(rawTraj_y, rawTraj_z, label="Raw")
    ax.plot(airTraj_y, airTraj_z, label="AirIMU")
    ax.plot(gt_y     , gt_z     , label="Ground Truth")
    
    ax.set_xlabel('Y axis')
    ax.set_ylabel('Z axis')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    full_path = os.path.join(category_path, "trajectory_yz.png")
    plt.savefig(full_path, dpi = 600)
    plt.close()
    
    ###########################################################
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    elevation_angle = 20  # Change the elevation angle (view from above/below)
    azimuthal_angle = 30  # Change the azimuthal angle (rotate around z-axis)

    ax.view_init(elevation_angle, azimuthal_angle)  # Set the view

    # Plotting the ground truth and inferred poses
    ax.plot(rawTraj_x, rawTraj_y, rawTraj_z, label="Raw")
    ax.plot(airTraj_x, airTraj_y, airTraj_z, label="AirIMU")
    ax.plot(gt_x     , gt_y     , gt_z     , label="Ground Truth")

    # Adding labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()

    full_path = os.path.join(category_path, "trajectory_3d.png")
    plt.savefig(full_path, dpi = 600)
    plt.close()


def visualize_velocity_with_uncertainty(
    save_prefix,
    gt_vel,
    air_vel,
    covs,
    dt,
    save_folder,
    mask=None,
    index_id=None,
):
    gt_vel = gt_vel.detach().cpu().numpy() if torch.is_tensor(gt_vel) else np.asarray(gt_vel)
    air_vel = air_vel.detach().cpu().numpy() if torch.is_tensor(air_vel) else np.asarray(air_vel)
    dt = dt.detach().cpu().numpy() if torch.is_tensor(dt) else np.asarray(dt)

    if gt_vel.ndim != 2 or air_vel.ndim != 2 or gt_vel.shape[1] != 3 or air_vel.shape[1] != 3:
        raise ValueError("Expected gt_vel and air_vel to have shape [N, 3].")

    seq_len = min(gt_vel.shape[0], air_vel.shape[0])
    gt_vel = gt_vel[:seq_len]
    air_vel = air_vel[:seq_len]

    if mask is not None:
        mask = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
        mask = mask.astype(bool)[:seq_len]
        gt_vel = gt_vel[mask]
        air_vel = air_vel[mask]

    if dt.ndim > 1:
        dt = dt.reshape(-1)

    if dt.size >= (gt_vel.shape[0] - 1):
        t = np.concatenate(([0.0], np.cumsum(dt[: gt_vel.shape[0] - 1])))
    else:
        t = np.arange(gt_vel.shape[0], dtype=np.float64)

    covs_np = covs.detach().cpu().numpy() if torch.is_tensor(covs) else np.asarray(covs)
    if covs_np.ndim == 4:
        covs_np = covs_np[0]
    if covs_np.ndim != 3 or covs_np.shape[-2:] != (9, 9):
        raise ValueError("Expected covs to have shape [K, 9, 9] or [1, K, 9, 9].")

    vel_var = np.clip(np.diagonal(covs_np, axis1=-2, axis2=-1)[:, 3:6], a_min=0.0, a_max=None)
    vel_sigma = np.sqrt(vel_var)

    if index_id is not None and dt.size > 0:
        idx = index_id.detach().cpu().numpy() if torch.is_tensor(index_id) else np.asarray(index_id)
        idx = idx.astype(np.int64)
        idx = np.concatenate(([0], idx))
        dt_cum = np.concatenate(([0.0], np.cumsum(dt)))
        idx = np.clip(idx, 0, dt_cum.shape[0] - 1)
        t_sigma = dt_cum[idx]
        if t_sigma.shape[0] != vel_sigma.shape[0]:
            t_sigma = np.linspace(0.0, t[-1] if t.size > 0 else 0.0, vel_sigma.shape[0])
    else:
        t_sigma = np.linspace(0.0, t[-1] if t.size > 0 else 0.0, vel_sigma.shape[0])

    axis_labels = ["x", "y", "z"]
    category_path = _category_dir(save_folder, save_prefix, "velocity")

    for axis in range(3):
        fig, (ax_vel, ax_unc) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10, 6),
            dpi=600,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )

        vel_gt_line = ax_vel.plot(t, gt_vel[:, axis], color="mediumseagreen", linewidth=1.0, label="GT velocity")[0]
        vel_air_line = ax_vel.plot(t, air_vel[:, axis], color="red", linewidth=1.0, label="AirIMU velocity")[0]
        sigma_line = ax_unc.plot(
            t_sigma,
            vel_sigma[:, axis],
            color="royalblue",
            linestyle="--",
            linewidth=1.0,
            label="AirIMU velocity uncertainty (1-sigma)",
        )[0]

        ax_vel.set_title(f"Velocity ({axis_labels[axis]}) vs Relative Time")
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_unc.set_xlabel("Relative time t (s)")
        ax_unc.set_ylabel("Uncertainty (m/s)")
        ax_vel.grid(True)
        ax_unc.grid(True)

        ax_vel.legend([vel_gt_line, vel_air_line], [vel_gt_line.get_label(), vel_air_line.get_label()], loc="best")
        ax_unc.legend([sigma_line], [sigma_line.get_label()], loc="best")

        full_path = os.path.join(category_path, f"velocity_{axis_labels[axis]}_with_uncertainty.png")
        plt.tight_layout()
        plt.savefig(full_path, dpi=600)
        plt.close(fig)


def box_plot_wrapper(ax, data, edge_color, fill_color, **kwargs):
    bp = ax.boxplot(data, **kwargs)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       
        
    return bp


def plot_boxes(folder, input_data, metrics, show_metrics):
    fig, ax = plt.subplots(dpi=300)
    raw_ticks   = [_-0.12 for _ in range(1, len(metrics) + 1)]
    air_ticks   = [_+0.12 for _ in range(1, len(metrics) + 1)]
    label_ticks = [_      for _ in range(1, len(metrics) + 1)]
    
    raw_data    = [input_data[metric + "(raw)"   ] for metric in metrics]
    air_data    = [input_data[metric + "(AirIMU)"] for metric in metrics]
    
    # ax.boxplot(data, patch_artist=True, positions=ticks, widths=.2)
    box_plot_wrapper(ax, raw_data, edge_color="black", fill_color="royalblue", positions=raw_ticks, patch_artist=True, widths=.2)
    box_plot_wrapper(ax, air_data, edge_color="black", fill_color="gold", positions=air_ticks, patch_artist=True, widths=.2)
    ax.set_xticks(label_ticks)
    ax.set_xticklabels(show_metrics)
    
    # Create color patches for legend
    gold_patch = mpatches.Patch(color='gold', label='AirIMU')
    royalblue_patch = mpatches.Patch(color='royalblue', label='Raw')
    ax.legend(handles=[gold_patch, royalblue_patch])
    
    plt.savefig(os.path.join(folder, "Metrics.png"), dpi = 600)
    plt.close()


def visualize_ave_barplot(all_results, save_folder, file_name="ave_raw_vs_airimu.png"):
    if all_results is None or len(all_results) == 0:
        return

    valid_results = [
        r for r in all_results
        if "name" in r and "AVE(raw)" in r and "AVE(AirIMU)" in r
    ]
    if len(valid_results) == 0:
        return

    flight_names = [str(r["name"]) for r in valid_results]
    ave_raw = [float(r["AVE(raw)"]) for r in valid_results]
    ave_airimu = [float(r["AVE(AirIMU)"]) for r in valid_results]

    x = np.arange(len(flight_names), dtype=np.float64)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(flight_names) * 1.1), 5), dpi=600)
    ax.bar(x - width / 2, ave_raw, width, color="red", label="Raw AVE")
    ax.bar(x + width / 2, ave_airimu, width, color="blue", label="AirIMU AVE")

    ax.set_title("AVE per Flight: Raw vs AirIMU")
    ax.set_ylabel("Average Velocity Error (m/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(flight_names, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best")

    summary_dir = os.path.join(save_folder, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, file_name), dpi=600)
    plt.close(fig)

