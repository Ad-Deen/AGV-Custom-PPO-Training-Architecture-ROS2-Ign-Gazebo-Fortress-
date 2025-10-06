# rl_race/set_pose_service.py

import rclpy
from rclpy.node import Node
from rl_race.srv import SetPose  # Make sure you import the service correctly

class SetPoseService(Node):

    def __init__(self):
        super().__init__('set_pose_service')
        self.srv = self.create_service(SetPose, 'set_pose', self.handle_set_pose)

    def handle_set_pose(self, request, response):
        # Implement your service logic here
        response.success = True  # Modify this as per your service logic
        return response

def main(args=None):
    rclpy.init(args=args)
    node = SetPoseService()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
