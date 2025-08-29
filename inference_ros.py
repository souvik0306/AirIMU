#!/usr/bin/env python3
import rospy
import numpy as np
import onnxruntime as ort
import pickle
import os
import rospkg
from sensor_msgs.msg import Imu
import sys

# --- configuration ---
SEQLEN   = 200
INTERVAL = 9
OVERLAP  = INTERVAL + 1  # number of samples kept between windows

class IMUBuffer:
    def __init__(self, seqlen, overlap):
        self.max_size = seqlen * 2
        self.seqlen = seqlen
        self.overlap = overlap
        self.time_buf = np.zeros(self.max_size, dtype=np.float32)
        self.acc_buf  = np.zeros((self.max_size, 3), dtype=np.float32)
        self.gyro_buf = np.zeros((self.max_size, 3), dtype=np.float32)
        self.buf_idx = 0

    def add(self, msg: Imu):
        self.time_buf[self.buf_idx] = msg.header.stamp.to_sec()
        self.acc_buf[self.buf_idx]  = [msg.linear_acceleration.x,
                                       msg.linear_acceleration.y,
                                       msg.linear_acceleration.z]
        self.gyro_buf[self.buf_idx] = [msg.angular_velocity.x,
                                       msg.angular_velocity.y,
                                       msg.angular_velocity.z]
        self.buf_idx += 1

    def ready(self):
        return self.buf_idx >= self.seqlen

    def get_window(self):
        return (self.time_buf[:self.buf_idx],
                self.acc_buf[:self.buf_idx],
                self.gyro_buf[:self.buf_idx])

    def slide_window(self):
        # Keep last overlap samples
        self.time_buf[:self.overlap] = self.time_buf[self.buf_idx - self.overlap : self.buf_idx]
        self.acc_buf[:self.overlap]  = self.acc_buf[self.buf_idx - self.overlap : self.buf_idx]
        self.gyro_buf[:self.overlap] = self.gyro_buf[self.buf_idx - self.overlap : self.buf_idx]
        self.buf_idx = self.overlap

class IMUInferenceNode:
    def __init__(self):
        rp = rospkg.RosPack()
        self.pkg_path = rp.get_path("imu_listener_pkg")
        self.onnx_path   = os.path.join(self.pkg_path, "models", "airimu_euroc.onnx")
        self.pickle_path = os.path.join(self.pkg_path, "results", "timeit_mh5_net_output.pickle")
        self.buffer = IMUBuffer(SEQLEN, OVERLAP)
        self.results = []
        self.correction_counter = 0
        self.onnx_model = None

    def check_files(self):
        if not os.path.isfile(self.onnx_path):
            rospy.logerr(f"ONNX model file not found: {self.onnx_path}")
            return False
        return True

    def load_model(self):
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = os.cpu_count()
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            self.onnx_model = ort.InferenceSession(
                self.onnx_path, sess_options=session_options, providers=["CPUExecutionProvider"]
            )
            rospy.loginfo(f"[READY] Model loaded at ROS time: {rospy.Time.now().to_sec():.2f}")
        except Exception as e:
            rospy.logerr(f"Failed to load ONNX model: {e}")
            rospy.signal_shutdown("Fatal error: Model loading failed.")

    def save_results(self):
        try:
            os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            rospy.logerr(f"Failed to save results: {e}")

    def run_inference(self):
        try:
            time, acc, gyro = self.buffer.get_window()
            dt = np.diff(time)[..., None]
            acc = acc[:-1]
            gyro = gyro[:-1]
            acc_b  = acc[None, ...]
            gyro_b = gyro[None, ...]
            corr_acc, corr_gyro = self.onnx_model.run(None, {"acc": acc_b, "gyro": gyro_b})

            start = OVERLAP - 1
            corrected_acc  = acc_b[:, start:, :]  + corr_acc
            corrected_gyro = gyro_b[:, start:, :] + corr_gyro
            dt_trim        = dt[start:, :]

            self.correction_counter += 1
            rospy.loginfo(f"[Correction #{self.correction_counter}]")
            rospy.loginfo(f"Corrected accel: {corrected_acc[0, -1]}")
            rospy.loginfo(f"Corrected gyro:  {corrected_gyro[0, -1]}")
            rospy.loginfo("-----------------------")

            self.results.append({
                "correction_acc":  corr_acc[0],
                "correction_gyro": corr_gyro[0],
                "corrected_acc":   corrected_acc[0],
                "corrected_gyro":  corrected_gyro[0],
                "dt":              dt_trim,
            })

            self.save_results()
        except Exception as e:
            rospy.logerr(f"Inference error: {e}")

    def imu_callback(self, msg: Imu):
        self.buffer.add(msg)
        if self.buffer.ready():
            self.run_inference()
            self.buffer.slide_window()

    def start(self):
        rospy.init_node("imu_inference_node")
        rospy.loginfo(f"[INIT] Node started at ROS time: {rospy.Time.now().to_sec():.2f}")
        rospy.loginfo(f"[INFO] Python version: {sys.version}")

        if not self.check_files():
            rospy.signal_shutdown("Required files missing.")
            return

        self.load_model()
        rospy.Subscriber("/imu0", Imu, self.imu_callback, queue_size=1000)
        # rospy.Subscriber("mavros/imu/data", Imu, self.imu_callback, queue_size=1000)
        rospy.spin()

if __name__ == "__main__":
    node = IMUInferenceNode()
    node.start()
