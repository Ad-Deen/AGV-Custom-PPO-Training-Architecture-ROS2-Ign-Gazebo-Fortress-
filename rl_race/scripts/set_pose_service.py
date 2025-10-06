#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from subprocess import run
from rl_race.srv import SetPose
import geometry_msgs.msg

class SetPoseService(Node):
    def __init__(self):
        super().__init__('set_pose_service')
        self.srv = self.create_service(SetPose, 'set_pose', self.set_pose_callback)

    def set_pose_callback(self, request, response):
        entity_name = request.entity_name
        pose = request.pose

        # Construct the command to call the Ignition service
        cmd = [
            'ign', 'service', '-s', '/world/empty_world/set_pose',
            '--reqtype', 'ignition.msgs.Pose',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '300',
            '--req', f'name: "{entity_name}", position: {{x: {pose.position.x}, y: {pose.position.y}, z: {pose.position.z}}}, orientation: {{x: {pose.orientation.x}, y: {pose.orientation.y}, z: {pose.orientation.z}, w: {pose.orientation.w}}}'
        ]

        # Run the command
        result = run(cmd, capture_output=True, text=True)

        # Check if the service call was successful
        response.success = 'data: true' in result.stdout

        return response

def main(args=None):
    rclpy.init(args=args)
    node = SetPoseService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
