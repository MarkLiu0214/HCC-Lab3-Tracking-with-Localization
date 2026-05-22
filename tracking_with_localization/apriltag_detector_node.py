#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import cv2
import numpy as np
import yaml
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray  # 新增 Marker 支援
from cv_bridge import CvBridge
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory

class AprilTagDetectorNode(Node):
    def __init__(self):
        super().__init__('apriltag_detector_node')
        self.bridge = CvBridge()
        
        # 相機參數與設定
        self.fx, self.fy = 907.45, 906.73
        self.cx, self.cy = 470.05, 369.95
        self.tag_size = 0.20
        self.camera_params = [self.fx, self.fy, self.cx, self.cy]
        
        # 初始化地圖與偵測器
        self.tag_map = {}
        self.detected_tags = set()
        self.load_tag_map()
        self.detector = Detector(families='tag36h11')
        
        # 定義發布者
        self.img_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.ego_pose_pub = self.create_publisher(PoseStamped, '/tello/ego_pose', 10)
        # 新增：發布畫框後的影像 [cite: 37, 48]
        self.image_pub = self.create_publisher(Image, '/tello/image_with_tags', 10)
        # 新增：發布地圖標籤到 RViz [cite: 33, 45]
        self.marker_pub = self.create_publisher(MarkerArray, '/apriltag/markers', 10)
        
        self.get_logger().info("多標籤定位節點已啟動，支援影像標記與 RViz 顯示。")

        self.timer = self.create_timer(0.1, self.publish_static_markers)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(5, self.keep_alive_callback)

    def keep_alive_callback(self):
        '''
        anti-sleeping
        '''
        msg = Twist()
        self.cmd_pub.publish(msg)
        self.get_logger().debug('Keep-alive cmd is sent')



    def load_tag_map(self):
        try:
            package_share_directory = get_package_share_directory('tracking_with_localization')
            yaml_path = os.path.join(package_share_directory, 'map', 'apriltag_map.yaml')
            
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                for tag in config['tags']:
                    T = np.eye(4)
                    T[:3, :3] = R.from_euler('xyz', tag['orientation_rpy']).as_matrix()
                    T[:3, 3] = tag['position']
                    self.tag_map[tag['id']] = T
            self.get_logger().info(f"成功加載地圖路徑: {yaml_path}")
        except Exception as e:
            self.get_logger().error(f"地圖載入失敗: {e}")

    def publish_static_markers(self):
            """ 常時發布地圖中的所有標籤，預設為紅色 """
            marker_array = MarkerArray()
            for tag_id, T_w_t in self.tag_map.items():
                marker = self.create_marker_msg(tag_id, T_w_t)
                
                # 如果該 ID 剛被偵測到，設為綠色，否則為紅色
                if tag_id in self.detected_tags:
                    marker.color.r, marker.color.g = 0.0, 1.0  # 綠色 
                else:
                    marker.color.r, marker.color.g = 1.0, 0.0  # 紅色
                
                marker_array.markers.append(marker)
            
            self.marker_pub.publish(marker_array)
            # 發布後清除偵測紀錄，等待下一影格更新
            self.detected_tags.clear()

    def create_marker_msg(self, tag_id, T_w_t):
        """ 封裝 Marker 建立邏輯 """
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = tag_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(T_w_t[0, 3])
        marker.pose.position.y = float(T_w_t[1, 3])
        marker.pose.position.z = float(T_w_t[2, 3])
        q = R.from_matrix(T_w_t[:3, :3]).as_quat()
        marker.pose.orientation.x, marker.pose.orientation.y = q[0], q[1]
        marker.pose.orientation.z, marker.pose.orientation.w = q[2], q[3]
        marker.scale.x, marker.scale.y, marker.scale.z = self.tag_size, self.tag_size, 0.01
        marker.color.a = 0.8
        
        return marker
        


    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        tags = self.detector.detect(gray, estimate_tag_pose=True, 
                                    camera_params=self.camera_params, 
                                    tag_size=self.tag_size)
        
        marker_array = MarkerArray()
        valid_poses = []

        if tags:
            for tag in tags:

                # Add detected list
                self.detected_tags.add(tag.tag_id)
        

                # --- 1. 在影像上畫出標籤外框 (OpenCV) ---
                corners = tag.corners.astype(np.int32)
                # 畫出四邊形外框
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                # 標示 ID 數字
                cv2.putText(frame, f"ID: {tag.tag_id}", (corners[0][0], corners[0][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if tag.tag_id not in self.tag_map:
                    continue

                # --- 2. 座標計算 (定位) ---
                T_c_t = np.eye(4)
                T_c_t[:3, :3] = tag.pose_R
                T_c_t[:3, 3] = tag.pose_t.flatten()
                
                T_w_t = self.tag_map[tag.tag_id]
                T_w_c_opencv = T_w_t @ np.linalg.inv(T_c_t)
                
                # 座標轉換矩陣：OpenCV 轉 ROS [cite: 27, 31]
                R_cv_to_ros = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
                T_w_c_ros = T_w_c_opencv @ R_cv_to_ros
                valid_poses.append(T_w_c_ros)

                # --- 3. 建立 RViz Marker ---
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = tag.tag_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = float(T_w_t[0, 3])
                marker.pose.position.y = float(T_w_t[1, 3])
                marker.pose.position.z = float(T_w_t[2, 3])
                
                q = R.from_matrix(T_w_t[:3, :3]).as_quat()
                marker.pose.orientation.x, marker.pose.orientation.y = q[0], q[1]
                marker.pose.orientation.z, marker.pose.orientation.w = q[2], q[3]
                
                marker.scale.x, marker.scale.y, marker.scale.z = self.tag_size, self.tag_size, 0.01
                marker.color.a, marker.color.g = 1.0, 1.0  # 綠色標籤 [cite: 33]
                marker_array.markers.append(marker)

            # 發布 RViz Marker
            self.marker_pub.publish(marker_array)

            # 發布 Ego Pose
            if valid_poses:
                final_T = valid_poses[0]
                pose = PoseStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = 'map'
                pose.pose.position.x = float(final_T[0, 3])
                pose.pose.position.y = float(final_T[1, 3])
                pose.pose.position.z = float(final_T[2, 3])
                q_ego = R.from_matrix(final_T[:3, :3]).as_quat()
                pose.pose.orientation.x, pose.pose.orientation.y = q_ego[0], q_ego[1]
                pose.pose.orientation.z, pose.pose.orientation.w = q_ego[2], q_ego[3]
                self.ego_pose_pub.publish(pose)

        # 發布畫好的影像
        img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        img_msg.header = msg.header
        self.image_pub.publish(img_msg)

def main():
    rclpy.init()
    rclpy.spin(AprilTagDetectorNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
