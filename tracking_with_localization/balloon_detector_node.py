#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped  # 修改：改用 PoseStamped [cite: 37]
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

class BalloonDetectorNode(Node):
    def __init__(self):
        super().__init__('balloon_detector_node')
        self.bridge = CvBridge()
        
        # 1. 相機內參 (需與 AprilTag 節點一致) [cite: 48, 57]
        self.fx, self.fy = 907.45, 906.73
        self.cx, self.cy = 470.05, 369.95
        self.balloon_real_size = 0.20  # 氣球直徑 (公尺)
        
        # 2. 狀態變數
        self.current_ego_pose = None
        
        # 3. 定義訂閱者與發布者
        self.img_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.ego_sub = self.create_subscription(PoseStamped, '/tello/ego_pose', self.ego_pose_callback, 10)
        
        # 修改：不再發布 Marker，而是發布 PoseStamped 給追蹤節點使用 [cite: 62]
        self.balloon_pose_pub = self.create_publisher(PoseStamped, '/balloon/pose_raw', 10)
        self.get_logger().info("紅氣球偵測節點已啟動，將發布原始位姿數據。")

    def ego_pose_callback(self, msg):
        self.current_ego_pose = msg

    def image_callback(self, msg):
        if self.current_ego_pose is None:
            self.get_logger().warn("尚未收到 Ego Pose，無法計算世界座標。", throttle_duration_sec=2.0)
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 4. 紅色遮罩處理 [cite: 44]
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)

            if area > 200:
                # 5. 計算 2D 像素位置與距離 [cite: 22, 61]
                M = cv2.moments(max_contour)
                if M['m00'] == 0: return
                u = int(M['m10'] / M['m00'])
                v = int(M['m01'] / M['m00'])
                
                pixel_diameter = 2.0 * math.sqrt(area / math.pi)
                Z_c = (self.fx * self.balloon_real_size) / pixel_diameter 
                
                X_c = ((u - self.cx) / self.fx) * Z_c
                Y_c = ((v - self.cy) / self.fy) * Z_c
                
                # 6. 座標轉換：Camera -> World (Map) [cite: 14, 22]
                self.publish_balloon_pose(X_c, Y_c, Z_c, msg.header.stamp)

    def publish_balloon_pose(self, x_c, y_c, z_c, timestamp):
        """ 將相機系下的球體座標轉到 map 系下並發布 PoseStamped """
        p = self.current_ego_pose.pose.position
        q = self.current_ego_pose.pose.orientation
        
        # 建立世界座標轉換矩陣
        T_w_c = np.eye(4)
        T_w_c[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T_w_c[:3, 3] = [p.x, p.y, p.z]
        
        # OpenCV -> ROS 座標系校正 
        # P_camera_ros = [Z(前), -X(左), -Y(上)]
        P_camera_ros = np.array([z_c, -x_c, -y_c, 1])
        P_world = T_w_c @ P_camera_ros
        
        # 修改：建立 PoseStamped 訊息 [cite: 62]
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = timestamp # 使用影像原始時間戳確保同步 
        pose_msg.pose.position.x = float(P_world[0])
        pose_msg.pose.position.y = float(P_world[1])
        pose_msg.pose.position.z = float(P_world[2])
        # 球體無方向性，設為單位四元數
        pose_msg.pose.orientation.w = 1.0
        
        self.balloon_pose_pub.publish(pose_msg)

def main():
    rclpy.init()
    rclpy.spin(BalloonDetectorNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
