#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import os
import cv2
import yaml
import numpy as np

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge
from pupil_apriltags import Detector

from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory


class AprilTagDetectorNode(Node):

    def __init__(self):

        super().__init__('apriltag_detector_node')

        self.bridge = CvBridge()

        # =====================================================
        # Camera Parameters
        # =====================================================
        self.fx = 907.45
        self.fy = 906.73
        self.cx = 470.05
        self.cy = 369.95

        self.tag_size = 0.20

        self.camera_params = [
            self.fx,
            self.fy,
            self.cx,
            self.cy
        ]

        # =====================================================
        # AprilTag Detector
        # =====================================================
        self.detector = Detector(
            families='tag36h11'
        )

        # =====================================================
        # Tag Map
        # =====================================================
        self.tag_map = {}

        self.detected_tags = set()

        self.load_tag_map()

        # =====================================================
        # Subscribers
        # =====================================================
        self.img_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        # =====================================================
        # Publishers
        # =====================================================
        self.ego_pose_pub = self.create_publisher(
            PoseStamped,
            '/tello/ego_pose',
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            '/tello/image_with_tags',
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/apriltag/markers',
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # =====================================================
        # Timers
        # =====================================================
        self.marker_timer = self.create_timer(
            0.1,
            self.publish_static_markers
        )

        self.keep_alive_timer = self.create_timer(
            5.0,
            self.keep_alive_callback
        )

        self.get_logger().info(
            'AprilTag detector node started.'
        )

    # =========================================================
    # Keep Alive
    # =========================================================
    def keep_alive_callback(self):

        msg = Twist()

        self.cmd_pub.publish(msg)

        self.get_logger().debug(
            'Keep-alive cmd is sent'
        )

    # =========================================================
    # Load Tag Map
    # =========================================================
    def load_tag_map(self):

        try:

            package_share_directory = \
                get_package_share_directory(
                    'tracking_with_localization'
                )

            yaml_path = os.path.join(
                package_share_directory,
                'map',
                'apriltag_map.yaml'
            )

            with open(yaml_path, 'r') as f:

                config = yaml.safe_load(f)

                for tag in config['tags']:

                    T = np.eye(4)

                    T[:3, :3] = R.from_euler(
                        'xyz',
                        tag['orientation_rpy']
                    ).as_matrix()

                    T[:3, 3] = tag['position']

                    self.tag_map[tag['id']] = T

            self.get_logger().info(
                f'Successfully loaded map: {yaml_path}'
            )

        except Exception as e:

            self.get_logger().error(
                f'Failed to load map: {e}'
            )

    # =========================================================
    # Publish Static Markers
    # =========================================================
    def publish_static_markers(self):

        marker_array = MarkerArray()

        for tag_id, T_w_t in self.tag_map.items():

            marker = self.create_marker_msg(
                tag_id,
                T_w_t
            )

            # Green if detected
            if tag_id in self.detected_tags:

                marker.color.r = 0.0
                marker.color.g = 1.0

            else:

                marker.color.r = 1.0
                marker.color.g = 0.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

        self.detected_tags.clear()

    # =========================================================
    # Create Marker
    # =========================================================
    def create_marker_msg(self, tag_id, T_w_t):

        marker = Marker()

        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.id = tag_id

        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = float(T_w_t[0, 3])
        marker.pose.position.y = float(T_w_t[1, 3])
        marker.pose.position.z = float(T_w_t[2, 3])

        q = R.from_matrix(
            T_w_t[:3, :3]
        ).as_quat()

        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        marker.scale.x = self.tag_size
        marker.scale.y = self.tag_size
        marker.scale.z = 0.01

        marker.color.a = 0.8

        return marker

    # =========================================================
    # Image Callback
    # =========================================================
    def image_callback(self, msg):

        # =====================================================
        # Convert Image
        # =====================================================
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            'bgr8'
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # =====================================================
        # Detect AprilTags
        # =====================================================
        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size
        )

        marker_array = MarkerArray()

        final_T = None

        # =====================================================
        # Find Largest Tag
        # =====================================================
        best_tag = None
        best_area = 0

        if tags:

            for tag in tags:

                corners = tag.corners.astype(np.int32)

                area = cv2.contourArea(corners)

                if area > best_area:

                    best_area = area
                    best_tag = tag

            # =================================================
            # Draw ALL Tags
            # =================================================
            for tag in tags:

                self.detected_tags.add(
                    tag.tag_id
                )

                corners = tag.corners.astype(
                    np.int32
                )

                cv2.polylines(
                    frame,
                    [corners],
                    True,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f'ID: {tag.tag_id}',
                    (
                        corners[0][0],
                        corners[0][1] - 10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # =================================================
            # Use Largest Tag For Localization
            # =================================================
            if best_tag is not None:

                tag = best_tag

                # Skip if tag not in map
                if tag.tag_id in self.tag_map:

                    # =========================================
                    # Camera -> Tag
                    # =========================================
                    T_c_t = np.eye(4)

                    T_c_t[:3, :3] = tag.pose_R

                    T_c_t[:3, 3] = \
                        tag.pose_t.flatten()

                    # =========================================
                    # World -> Tag
                    # =========================================
                    T_w_t = self.tag_map[
                        tag.tag_id
                    ]

                    # =========================================
                    # World -> Camera (OpenCV)
                    # =========================================
                    T_w_c_opencv = \
                        T_w_t @ np.linalg.inv(T_c_t)

                    # =========================================
                    # OpenCV -> ROS
                    # =========================================
                    R_cv_to_ros = np.array([
                        [0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]
                    ])

                    final_T = \
                        T_w_c_opencv @ R_cv_to_ros

                    # =========================================
                    # Highlight Selected Tag
                    # =========================================
                    corners = tag.corners.astype(
                        np.int32
                    )

                    cv2.polylines(
                        frame,
                        [corners],
                        True,
                        (255, 0, 0),
                        4
                    )

                    cv2.putText(
                        frame,
                        'BEST TAG',
                        (
                            corners[0][0],
                            corners[0][1] - 40
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2
                    )

        # =====================================================
        # Publish Marker Array
        # =====================================================
        self.marker_pub.publish(
            marker_array
        )

        # =====================================================
        # Publish Ego Pose
        # =====================================================
        if final_T is not None:

            pose = PoseStamped()

            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = 'map'

            pose.pose.position.x = \
                float(final_T[0, 3])

            pose.pose.position.y = \
                float(final_T[1, 3])

            pose.pose.position.z = \
                float(final_T[2, 3])

            q_ego = R.from_matrix(
                final_T[:3, :3]
            ).as_quat()

            pose.pose.orientation.x = q_ego[0]
            pose.pose.orientation.y = q_ego[1]
            pose.pose.orientation.z = q_ego[2]
            pose.pose.orientation.w = q_ego[3]

            self.ego_pose_pub.publish(
                pose
            )

        # =====================================================
        # Publish Image
        # =====================================================
        img_msg = self.bridge.cv2_to_imgmsg(
            frame,
            'bgr8'
        )

        img_msg.header = msg.header

        self.image_pub.publish(
            img_msg
        )


# =============================================================
# Main
# =============================================================
def main():

    rclpy.init()

    node = AprilTagDetectorNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
