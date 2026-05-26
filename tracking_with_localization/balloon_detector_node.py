#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

class BalloonDetectorNode(Node):
    def __init__(self):
        super().__init__('balloon_detector_node')
        self.bridge = CvBridge()
        
        # 1. 相機內參與氣球設定 [cite: 48, 57]
        self.fx, self.fy = 907.45, 906.73
        self.cx, self.cy = 470.05, 369.95
        self.balloon_real_size = 0.20 
        
        # 儲存來自 AprilTag 節點處理過的影像
        self.image_with_tags = None
        self.current_ego_pose = None
        
        # 2. 定義訂閱者
        # 訂閱原始影像用於偵測運算
        self.img_raw_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        # 訂閱已標記 AprilTag 的影像用於繪圖 [cite: 235, 236]
        self.img_tags_sub = self.create_subscription(Image, '/tello/image_with_tags', self.tags_image_callback, 10)
        # 訂閱無人機位姿
        self.ego_sub = self.create_subscription(PoseStamped, '/tello/ego_pose', self.ego_pose_callback, 10)
        
        # 3. 定義發布者
        self.balloon_pose_pub = self.create_publisher(PoseStamped, '/balloon/pose_raw', 10)
        # 發布同時包含 AprilTag 與紅球框的最終影像 
        self.image_all_pub = self.create_publisher(Image, '/tello/image_with_all_detections', 10)
        
        self.get_logger().info("紅氣球偵測節點已啟動，影像流水線已建立。")

    def ego_pose_callback(self, msg):
        self.current_ego_pose = msg

    def tags_image_callback(self, msg):
        """ 接收來自 AprilTag 節點標記過的影像 """
        self.image_with_tags = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def image_callback(self, msg):
            """ 使用原始影像進行偵測邏輯 """
            # 如果還沒收到無人機位姿，暫不處理偵測，但可以考慮發布空畫面
            has_pose = self.current_ego_pose is not None

            # 1. 轉換原始影像進行 HSV 偵測 
            raw_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            hsv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)

            # --- 重要修改：初始化底圖 --- 
            # 無論有沒有紅球，我們都要準備一張底圖發布。
            # 優先使用帶有 AprilTag 框的影像，若無則使用原始影像。
            if self.image_with_tags is not None:
                display_frame = self.image_with_tags.copy()
            else:
                display_frame = raw_frame.copy()

            # 2. 紅色遮罩處理
            lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(max_contour)

                if area > 200:
                    # 3. 計算 3D 位置與發布 Pose
                    M = cv2.moments(max_contour)
                    if M['m00'] != 0:
                        u, v = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                        pixel_diameter = 2.0 * math.sqrt(area / math.pi)
                        Z_c = (self.fx * self.balloon_real_size) / pixel_diameter 
                        X_c = ((u - self.cx) / self.fx) * Z_c
                        Y_c = ((v - self.cy) / self.fy) * Z_c
                        
                                # 只有有 ego pose 時才發布世界座標
                        if self.current_ego_pose is not None:
                            self.publish_balloon_pose(
                                X_c, Y_c, Z_c, msg.header.stamp
                            )

                        # --- 4. 在底圖上畫紅球框 --- [cite: 255]
                        rx, ry, rw, rh = cv2.boundingRect(max_contour)
                        cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 3)
                        cv2.putText(display_frame, "Balloon", (rx, ry - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # --- 5. 核心修復：發布最終影像 (移出所有 if 條件) --- [cite: 275]
            # 這樣不論：(1)有無 AprilTag (2)有無紅球，RViz 都會有畫面
            combined_msg = self.bridge.cv2_to_imgmsg(display_frame, 'bgr8')
            combined_msg.header = msg.header
            self.image_all_pub.publish(combined_msg)

    def publish_balloon_pose(self, x_c, y_c, z_c, timestamp):
        p, q = self.current_ego_pose.pose.position, self.current_ego_pose.pose.orientation
        T_w_c = np.eye(4)
        T_w_c[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T_w_c[:3, 3] = [p.x, p.y, p.z]
        
        P_camera_ros = np.array([z_c, -x_c, -y_c, 1])
        P_world = T_w_c @ P_camera_ros
        
        pose_msg = PoseStamped()
        pose_msg.header.frame_id, pose_msg.header.stamp = "map", timestamp
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = float(P_world[0]), float(P_world[1]), float(P_world[2])
        pose_msg.pose.orientation.w = 1.0
        self.balloon_pose_pub.publish(pose_msg)

def main():
    rclpy.init()
    rclpy.spin(BalloonDetectorNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
