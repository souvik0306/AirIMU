import os
import torch
import numpy as np
import pypose as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _safe_drive_parts(save_prefix):
    parts = []
    for part in str(save_prefix).replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        parts.append(part.replace("..", "__"))
    return parts if parts else ["sequence"]


def _category_dir(save_folder, save_prefix, category):
    category_path = os.path.join(save_folder, *_safe_drive_parts(save_prefix), category)
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


def visualize_velocity_with_model_uncertainty(
    save_prefix,
    gt_vel,
    air_vel,
    acc_cov,
    gyro_cov,
    dt,
    save_folder,
    mask=None,
):
    gt_vel = gt_vel.detach().cpu().numpy() if torch.is_tensor(gt_vel) else np.asarray(gt_vel)
    air_vel = air_vel.detach().cpu().numpy() if torch.is_tensor(air_vel) else np.asarray(air_vel)
    acc_cov = acc_cov.detach().cpu().numpy() if torch.is_tensor(acc_cov) else np.asarray(acc_cov)
    gyro_cov = gyro_cov.detach().cpu().numpy() if torch.is_tensor(gyro_cov) else np.asarray(gyro_cov)
    dt = dt.detach().cpu().numpy() if torch.is_tensor(dt) else np.asarray(dt)

    if gt_vel.ndim != 2 or air_vel.ndim != 2 or gt_vel.shape[1] != 3 or air_vel.shape[1] != 3:
        raise ValueError("Expected gt_vel and air_vel to have shape [N, 3].")
    if acc_cov.ndim == 3:
        acc_cov = acc_cov[0]
    if gyro_cov.ndim == 3:
        gyro_cov = gyro_cov[0]
    if acc_cov.ndim != 2 or gyro_cov.ndim != 2 or acc_cov.shape[1] != 3 or gyro_cov.shape[1] != 3:
        raise ValueError("Expected acc_cov and gyro_cov to have shape [N, 3] or [1, N, 3].")

    seq_len = min(gt_vel.shape[0], air_vel.shape[0], acc_cov.shape[0], gyro_cov.shape[0])
    gt_vel = gt_vel[:seq_len]
    air_vel = air_vel[:seq_len]
    acc_cov = acc_cov[:seq_len]
    gyro_cov = gyro_cov[:seq_len]

    if dt.ndim > 1:
        dt = dt.reshape(-1)
    if dt.size >= seq_len:
        t = np.concatenate(([0.0], np.cumsum(dt[: seq_len - 1])))
    else:
        t = np.arange(seq_len, dtype=np.float64)

    if mask is not None:
        mask = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
        mask = mask.astype(bool)[:seq_len]
        t = t[mask]
        gt_vel = gt_vel[mask]
        air_vel = air_vel[mask]
        acc_cov = acc_cov[mask]
        gyro_cov = gyro_cov[mask]

    axis_labels = ["x", "y", "z"]
    category_path = _category_dir(save_folder, save_prefix, "uncertainty")

    for axis, axis_label in enumerate(axis_labels):
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 9), dpi=600)

        axs[0].plot(t, gt_vel[:, axis], color="mediumseagreen", linewidth=0.9, label=f"GT vel_{axis_label}")
        axs[0].plot(t, air_vel[:, axis], color="red", linewidth=0.9, label=f"AirIMU vel_{axis_label}")
        axs[1].plot(t, acc_cov[:, axis], color="tab:orange", linewidth=0.9, label=f"acc_cov_{axis_label}")
        axs[2].plot(t, gyro_cov[:, axis], color="tab:purple", linewidth=0.9, label=f"gyro_cov_{axis_label}")

        axs[0].set_title(f"Velocity and Model Uncertainty Outputs ({axis_label} axis)")
        axs[0].set_ylabel("velocity (m/s)")
        axs[1].set_ylabel("acc_cov output")
        axs[2].set_ylabel("gyro_cov output")
        axs[2].set_xlabel("Relative time t (s)")

        for ax in axs:
            ax.grid(True)
            ax.legend(loc="best")

        plt.tight_layout()
        plt.savefig(
            os.path.join(category_path, f"velocity_{axis_label}_with_uncertainty_outputs.png"),
            dpi=600,
        )
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
