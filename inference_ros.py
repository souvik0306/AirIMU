#!/usr/bin/env python3
"""ROS node running AirIMU ONNX model and saving results for evaluation."""
import os
import pickle
import numpy as np
import torch
import onnxruntime as ort
import rospy
from sensor_msgs.msg import Imu

# --- configuration ---
SEQLEN = 200               # number of samples per inference call (>=10)
INTERVAL = 9               # model context size

# parameters populated after init_node
SEQ_NAME: str
ONNX_PATH: str
PICKLE_PATH: str
onnx_model: ort.InferenceSession

# buffers and result accumulators
_time_buf, _acc_buf, _gyro_buf = [], [], []
results = {
    "correction_acc": [],
    "correction_gyro": [],
    "corrected_acc": [],
    "corrected_gyro": [],
    "dt": [],
}

def _save_results() -> None:
    out = {SEQ_NAME: {k: torch.cat(v, dim=0) for k, v in results.items()}}
    os.makedirs(os.path.dirname(PICKLE_PATH) or '.', exist_ok=True)
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

def run_inference() -> None:
    """Prepare tensors, run ONNX model and store results."""
    time = np.asarray(_time_buf, dtype=np.float64)
    acc = np.asarray(_acc_buf, dtype=np.float32)
    gyro = np.asarray(_gyro_buf, dtype=np.float32)

    dt = np.diff(time)[..., None]
    acc = acc[:-1]
    gyro = gyro[:-1]

    acc_b = acc[None, ...]
    gyro_b = gyro[None, ...]

    corr_acc, corr_gyro = onnx_model.run(None, {"acc": acc_b, "gyro": gyro_b})

    corrected_acc = acc_b[:, INTERVAL:, :] + corr_acc
    corrected_gyro = gyro_b[:, INTERVAL:, :] + corr_gyro
    dt_trim = dt[INTERVAL:, :]

    rospy.loginfo(f"Corrected accel: {corrected_acc[0, -1]}")
    rospy.loginfo(f"Corrected gyro:  {corrected_gyro[0, -1]}")
    rospy.loginfo("-----------------------")

    results["correction_acc"].append(torch.from_numpy(corr_acc[0]).to(dtype=torch.float64))
    results["correction_gyro"].append(torch.from_numpy(corr_gyro[0]).to(dtype=torch.float64))
    results["corrected_acc"].append(torch.from_numpy(corrected_acc[0]).to(dtype=torch.float64))
    results["corrected_gyro"].append(torch.from_numpy(corrected_gyro[0]).to(dtype=torch.float64))
    results["dt"].append(torch.tensor(dt_trim, dtype=torch.float64))

    _save_results()

def imu_callback(msg: Imu) -> None:
    t = msg.header.stamp.to_sec()
    acc = (msg.linear_acceleration.x,
           msg.linear_acceleration.y,
           msg.linear_acceleration.z)
    gyro = (msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z)

    _time_buf.append(t)
    _acc_buf.append(acc)
    _gyro_buf.append(gyro)

    if len(_time_buf) >= SEQLEN:
        run_inference()
        _time_buf[:] = _time_buf[-INTERVAL:]
        _acc_buf[:] = _acc_buf[-INTERVAL:]
        _gyro_buf[:] = _gyro_buf[-INTERVAL:]


def listener() -> None:
    """Initialize ROS node and start IMU subscription."""
    global SEQ_NAME, ONNX_PATH, PICKLE_PATH, onnx_model
    rospy.init_node("imu_inference_node")
    SEQ_NAME = rospy.get_param("~seq_name", "rosbag")
    ONNX_PATH = rospy.get_param("~onnx", "airimu_euroc.onnx")
    PICKLE_PATH = rospy.get_param("~outfile", "net_output.pickle")
    onnx_model = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    rospy.Subscriber("/imu0", Imu, imu_callback, queue_size=1000)
    rospy.spin()


if __name__ == '__main__':
    listener()
