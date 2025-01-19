#!/usr/bin/python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
import tf

class SLAMWithRRT:
    def __init__(self):
        rospy.init_node('slam_rrt_node')

        # Parameters
        self.map_width = rospy.get_param('~map_width', 100)
        self.map_height = rospy.get_param('~map_height', 100)
        self.resolution = rospy.get_param('~resolution', 0.05)

        # Initialize occupancy grid
        self.occupancy_grid = np.zeros((self.map_height, self.map_width), dtype=int)

        # Publishers and Subscribers
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)

        # TF Broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Rate
        self.rate = rospy.Rate(10)

    def laser_callback(self, scan_data):
        """Process laser scan data and update the map."""
        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))
        points = [
            (r * np.cos(angle), r * np.sin(angle))
            for r, angle in zip(ranges, angles) if scan_data.range_min < r < scan_data.range_max
        ]
        self.update_map(points)

    def update_map(self, points):
        """Update occupancy grid with detected points."""
        for point in points:
            x, y = point
            map_x = int((x + self.map_width * self.resolution / 2) / self.resolution)
            map_y = int((y + self.map_height * self.resolution / 2) / self.resolution)
            if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                self.occupancy_grid[map_y, map_x] = 100
        self.publish_map()

    def publish_map(self):
        """Publish the occupancy grid to the /map topic."""
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map"
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.map_width
        grid_msg.info.height = self.map_height
        grid_msg.info.origin.position.x = -self.map_width * self.resolution / 2
        grid_msg.info.origin.position.y = -self.map_height * self.resolution / 2
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = self.occupancy_grid.flatten().tolist()
        self.map_pub.publish(grid_msg)

    def publish_transform(self):
        """Broadcast the static transform from map to odom."""
        self.tf_broadcaster.sendTransform(
            (0, 0, 0),  # Translation
            tf.transformations.quaternion_from_euler(0, 0, 0),  # Rotation
            rospy.Time.now(),
            "odom",  # Child frame
            "map"     # Parent frame
        )

    def run(self):
        """Main loop for the SLAM node."""
        while not rospy.is_shutdown():
            self.publish_transform()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        slam_rrt = SLAMWithRRT()
        slam_rrt.run()
    except rospy.ROSInterruptException:
        pass
