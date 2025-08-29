#!/usr/bin/env python3
import rospy
import numpy as np
import onnxruntime as ort
import pickle
import os
import rospkg
from sensor_msgs.msg import Imu

# --- configuration ---
SEQLEN   = 200
INTERVAL = 9
OVERLAP  = INTERVAL + 1  # number of samples kept between windows

# resolve absolute paths using rospkg
rp = rospkg.RosPack()
pkg_path = rp.get_path("imu_listener_pkg")  # ← your package name

ONNX_PATH   = os.path.join(pkg_path, "models", "airimu_euroc.onnx")
PICKLE_PATH = os.path.join(pkg_path, "results", "timeit_mh_net_output.pickle")

# Load ONNX model
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = os.cpu_count()
session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

onnx_model = ort.InferenceSession(
    ONNX_PATH, sess_options=session_options, providers=["CPUExecutionProvider"]
)
# onnx_model = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# buffers
MAX_BUF_SIZE = SEQLEN * 2  # or any safe upper bound
time_buf = np.zeros(MAX_BUF_SIZE, dtype=np.float32)
acc_buf = np.zeros((MAX_BUF_SIZE, 3), dtype=np.float32)
gyro_buf = np.zeros((MAX_BUF_SIZE, 3), dtype=np.float32)
buf_idx = 0
results = []

def run_inference():
    # Use slices instead of .copy() to avoid extra memory usage
    time = time_buf[:buf_idx]
    acc  = acc_buf[:buf_idx]
    gyro = gyro_buf[:buf_idx]

    # compute dt and drop the last sample to match its length
    dt = np.diff(time)[..., None]
    acc = acc[:-1]
    gyro = gyro[:-1]

    acc_b  = acc[None, ...]
    gyro_b = gyro[None, ...]

    corr_acc, corr_gyro = onnx_model.run(None, {"acc": acc_b, "gyro": gyro_b})

    # skip the already-corrected prefix of length OVERLAP-1
    start = OVERLAP - 1
    corrected_acc  = acc_b[:, start:, :]  + corr_acc
    corrected_gyro = gyro_b[:, start:, :] + corr_gyro
    dt_trim        = dt[start:, :]

    rospy.loginfo(f"Corrected accel: {corrected_acc[0, -1]}")
    rospy.loginfo(f"Corrected gyro:  {corrected_gyro[0, -1]}")
    rospy.loginfo("-----------------------")

    results.append({
        "correction_acc":  corr_acc[0],
        "correction_gyro": corr_gyro[0],
        "corrected_acc":   corrected_acc[0],
        "corrected_gyro":  corrected_gyro[0],
        "dt":              dt_trim,
    })

    # save results
    os.makedirs(os.path.dirname(PICKLE_PATH), exist_ok=True)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def imu_callback(msg: Imu):
    global buf_idx
    time_buf[buf_idx] = msg.header.stamp.to_sec()
    acc_buf[buf_idx] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
    gyro_buf[buf_idx] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
    buf_idx += 1

    if buf_idx >= SEQLEN:
        run_inference()
        # Keep last OVERLAP samples
        time_buf[:OVERLAP] = time_buf[buf_idx-OVERLAP:buf_idx]
        acc_buf[:OVERLAP] = acc_buf[buf_idx-OVERLAP:buf_idx]
        gyro_buf[:OVERLAP] = gyro_buf[buf_idx-OVERLAP:buf_idx]
        buf_idx = OVERLAP

def listener():
    rospy.init_node("imu_inference_node")
    rospy.Subscriber("/imu0", Imu, imu_callback, queue_size=1000)
    rospy.spin()

if __name__ == "__main__":
    listener()