#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os
import csv
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

class BalloonTrackingKFNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        
        # 1. Kalman Filter 初始化 (等速模型 Constant Velocity)
        # 狀態向量 x = [x, y, z, vx, vy, vz]^T 
        self.dt = 0.1  # 預期更新頻率 (秒)
        self.x = np.zeros((6, 1)) 
        self.P = np.eye(6) * 1.0  # 估計誤差協方差
        self.F = np.eye(6)        # 狀態轉移矩陣 
        self.H = np.zeros((3, 6)) # 觀測矩陣
        self.H[:3, :3] = np.eye(3)
        
        # 雜訊矩陣設定
        self.Q = np.eye(6) * 0.01 # 過程雜訊
        self.R = np.eye(3) * 0.1  # 測量雜訊 
        
        self.is_initialized = False
        self.last_time = self.get_clock().now()

        # 2. 軌跡紀錄與 RViz Path 
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'map' #[cite: 58]
        self.trajectory_data = [] # 用於存入 CSV 

        # 3. 訂閱者與發布者
        # 訂閱來自 balloon_detector 的原始 3D 位置 [cite: 63]
        self.sub_pose = self.create_subscription(PoseStamped, '/balloon/pose_raw', self.pose_callback, 10)
        
        # 發布者：KF 平滑後的結果、預測位置、軌跡 [cite: 13, 53]
        self.marker_pub = self.create_publisher(MarkerArray, '/balloon/tracking_markers', 10)
        self.path_pub = self.create_publisher(Path, '/balloon/trajectory', 10)

        self.get_logger().info("Kalman Filter 軌跡追蹤節點已啟動。")

    def pose_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0: dt = self.dt
        
        # 更新狀態轉移矩陣 F 
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        
        # 取得觀測值 z = [x, y, z]
        z = np.array([[msg.pose.position.x], 
                      [msg.pose.position.y], 
                      [msg.pose.position.z]])

        if not self.is_initialized:
            self.x[:3] = z
            self.is_initialized = True
            self.last_time = current_time
            return

        # --- Kalman Filter 預測步 (Predict) ---
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # --- Kalman Filter 更新步 (Update) ---
        y = z - self.H @ self.x  # 測量殘差
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S) # 卡爾曼增益
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        self.last_time = current_time

        # --- 軌跡處理與發布 ---
        self.update_trajectory_and_publish(msg.header.stamp)

    def update_trajectory_and_publish(self, stamp):
        # 1. 紀錄 Path 軌跡 [cite: 53, 58]
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = stamp
        pose.pose.position.x = float(self.x[0])
        pose.pose.position.y = float(self.x[1])
        pose.pose.position.z = float(self.x[2])
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

        # 2. 紀錄資料用於 CSV 
        self.trajectory_data.append([stamp.sec + stamp.nanosec*1e-9, self.x[0,0], self.x[1,0], self.x[2,0]])

        # 3. 預測 0.5 秒後的位置 
        predict_time = 0.5
        x_pred = self.x[0,0] + self.x[3,0] * predict_time
        y_pred = self.x[1,0] + self.x[4,0] * predict_time
        z_pred = self.x[2,0] + self.x[5,0] * predict_time

        # 4. 建立 MarkerArray 發布給 RViz [cite: 13]
        markers = MarkerArray()
        # 目前位置 (KF平滑後) - 紅色
        markers.markers.append(self.create_sphere(1, [self.x[0,0], self.x[1,0], self.x[2,0]], [1.0, 0.0, 0.0, 1.0]))
        # 預測位置 (0.5s) - 黃色半透明 
        markers.markers.append(self.create_sphere(2, [x_pred, y_pred, z_pred], [1.0, 1.0, 0.0, 0.5]))
        
        self.marker_pub.publish(markers)

    def create_sphere(self, id, pos, color):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = pos
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    def destroy_node(self):
        # 節點關閉時儲存 CSV 
        save_path = os.path.expanduser('~/balloon_kf_trajectory.csv')
        try:
            with open(save_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'X', 'Y', 'Z'])
                writer.writerows(self.trajectory_data)
            self.get_logger().info(f'軌跡已儲存至: {save_path}')
        except Exception as e:
            self.get_logger().error(f'儲存失敗: {e}')
        super().destroy_node()

def main():
    rclpy.init()
    node = BalloonTrackingKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
