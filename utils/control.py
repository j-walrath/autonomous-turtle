from math import floor
import numpy as np
from numpy.linalg import norm
import pybullet as pb
from utils import simulator_library as lib
from utils.rvo.kdtree import KdTree
import logging


DEG_TO_RAD = np.pi / 180
MAX_LINEAR_V = 3.5
MIN_LINEAR_V = -1.5
MAX_ANGULAR_V = 3.0
MIN_ANGULAR_V = -3.0


def curvature(r, theta, delta):
    # ratio of rate of change in theta to the rate of change in r
    k1 = 1.0

    # timescale factor between fast subsystem and slow manifold
    k2 = 10

    return -(1 / r) * (k2 * (delta - np.arctan(-k1 * theta)) + (1 + k1 / (1 + (k1 * theta) ** 2)) * np.sin(delta))


def compute_v(k, vmax):
    # higher beta = velocity drops more quickly as a function of k
    beta = 0.4

    # higher gamma = sharper peak for v/vmax vs k curve
    gamma = 1

    return vmax / (1 + (beta * abs(k) ** gamma))


# def get_rays(pose, length, height):
#     (x, y, yaw) = pose
#     sideLength = length * 0.6
#     factor = 1.5
#     rayFrom = []
#     rayTo = []
#
#     rayFrom.append([x * np.cos(yaw), y * np.sin(yaw), height])
#     rayTo.append([rayFrom[0][0] + length * np.cos(yaw), rayFrom[0][1] + length * np.sin(yaw), height])
#
#     for i in np.linspace(0, 2 * np.pi, 100):
#         rayFrom.append(rayFrom[-1])
#         rayTo.append([rayFrom[-1][0] + length * np.cos(yaw + i), rayFrom[-1][1] + length * np.sin(yaw + i), height])
#
#     return rayFrom, rayTo


class RobotControl:
    minimum_linear_velocity = 0
    critical_distance = .5
    activation_threshold = 15 * DEG_TO_RAD
    object_distance_offset = 0.3

    def __init__(self, pb_client, robot_fsms, max_linear_velocity=MAX_LINEAR_V,
                 max_rotational_velocity=5.0):
        self.pb_client = pb_client
        self.robot_fsms = robot_fsms
        self.max_linear_velocity = max_linear_velocity
        self.max_rotational_velocity = max_rotational_velocity

    def get_object_state(self, object_id):
        position, orientation = self.pb_client.getBasePositionAndOrientation(bodyUniqueId=object_id)
        return position[0], position[1]

    def get_manipulator_state(self, robot_id):
        joint_angles, velocity, _, _ = zip(
            *self.pb_client.getJointStates(bodyUniqueId=robot_id, jointIndices=range(11 - 4, 15 - 4)))

        return joint_angles, velocity

    def get_robot_state(self, robot_id):
        position, orientation = self.pb_client.getBasePositionAndOrientation(bodyUniqueId=robot_id)
        _, _, yaw = pb.getEulerFromQuaternion(orientation)
        pose = (position[0], position[1], yaw)

        linear_velocity, angular_velocity = self.pb_client.getBaseVelocity(bodyUniqueId=robot_id)
        velocity = (linear_velocity[0], linear_velocity[1], angular_velocity[2])

        return pose, velocity

    def gripper_control(self, robot_id, gripper_target_state):
        for i, value in enumerate(gripper_target_state):
            self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=15 + i,
                                                 controlMode=pb.POSITION_CONTROL, targetPosition=value)

    def manipulator_control(self, robot_id, arm_target_state):
        for i, value in enumerate(arm_target_state):
            self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=11 + i - 4,
                                                 controlMode=pb.POSITION_CONTROL, targetPosition=value,
                                                 maxVelocity=5.0 * 2)

    def gripper_pinch(self, robot_id):
        gripper_target_state = (-0.016812667692005963, -0.01000778853017392)

        self.gripper_control(robot_id, gripper_target_state)

    def gripper_release(self, robot_id):
        gripper_target_state = (0.01786004943591301, 0.021918541425825912)

        self.gripper_control(robot_id, gripper_target_state)

    def manipulator_origin(self, robot_id):
        arm_target_state = (DEG_TO_RAD * 0, DEG_TO_RAD * -80, DEG_TO_RAD * 60, DEG_TO_RAD * 30)

        self.manipulator_control(robot_id, arm_target_state)

    def manipulator_lower(self, robot_id):
        arm_target_state = (-0.0016962233862355199, 1.2404879177129509, -0.901944524850455, 1.1624811955078364)

        self.manipulator_control(robot_id, arm_target_state)

    def manipulator_place(self, robot_id):
        arm_target_state = (DEG_TO_RAD * 180 - .2 * np.random.random(), 0, .4, 1.1)

        self.manipulator_control(robot_id, arm_target_state)

    def velocity_control(self, robot_id, linear_velocity, rotational_velocity, r=0.033, gain_v=0.1):
        left_wheel_velocity = linear_velocity / r - rotational_velocity * 0.027135999999999997 / (.144 * r)
        right_wheel_velocity = linear_velocity / r + rotational_velocity * 0.027135999999999997 / (.144 * r)

        self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                             jointIndex=0,
                                             controlMode=pb.VELOCITY_CONTROL,
                                             targetVelocity=left_wheel_velocity,
                                             force=500,
                                             velocityGain=gain_v)
        self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                             jointIndex=1,
                                             controlMode=pb.VELOCITY_CONTROL,
                                             targetVelocity=right_wheel_velocity,
                                             force=500,
                                             velocityGain=gain_v)

        self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                             jointIndex=2,
                                             controlMode=pb.VELOCITY_CONTROL,
                                             targetVelocity=left_wheel_velocity,
                                             force=500,
                                             velocityGain=gain_v)
        self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                             jointIndex=3,
                                             controlMode=pb.VELOCITY_CONTROL,
                                             targetVelocity=right_wheel_velocity,
                                             force=500,
                                             velocityGain=gain_v)

    def pose_control(self, robot_id, destination):
        pose, v_current = self.get_robot_state(robot_id=robot_id)
        yaw = pose[2]

        # vector from robot to target
        r = np.array(destination) - np.array(pose[0:2])

        # orientation of target w.r.t. line of sight from robot to target (-pi, pi]
        theta = 0

        # orientation of robot w.r.t. line of sight (-pi, pi]
        delta = lib.normalize_angle(yaw - np.arctan2(r[1], r[0]))

        k = curvature(norm(r), theta, delta)
        v = compute_v(k, self.max_linear_velocity)
        w = k * v

        return v, w

    # Integration of pose controller with DD-ORCA
    def smart_pose_control(self, robot_id, destination):
        pose, v_current = self.get_robot_state(robot_id)
        v, w = self.pose_control(robot_id, destination)

        # rays_from, rays_to = get_rays(pose, 5.0, 0.055)
        # ray_results = pb.rayTestBatch(rays_from, rays_to)

        # Find agents, obstacle_robots
        moving_robots = []
        obstacle_robots = []
        for ID in self.robot_fsms:
            if ID != robot_id:
                if self.robot_fsms[ID].current_state not in ("MOVE", "RETRIEVE"):
                    obstacle_robots.append(ID)
                else:
                    moving_robots.append(ID)

        if not (moving_robots or obstacle_robots):
            # No collision avoidance necessary if there is only one robot on the field.
            self.velocity_control(robot_id, linear_velocity=v, rotational_velocity=w)
            return v, w
        else:
            # Compute agent p, v_eff, v_pref
            agent = lib.get_agent(robot_id, pose=pose, v=v_current)
            agent.pref_velocity_ = lib.get_vec(lib.convert_v(pose, v, w))

            # Compute p, v_eff for all neighbors
            other_agents = []
            for other_robot in moving_robots:
                agent_pose, agent_v = self.get_robot_state(other_robot)
                other_agents.append(lib.get_agent(other_robot, agent_pose, agent_v))

            # Add non-moving robots to the obstacle list.
            obstacles = lib.get_obstacle_list([self.get_robot_state(obstacle)[0] for obstacle in obstacle_robots])

            # Build KD Trees for agents and obstacles
            agent.kd_tree_ = lib.build_trees(other_agents + [agent], obstacles)

            # Use ORCA to compute adjusted velocity
            agent.compute_neighbors()
            agent.compute_new_velocity()

            # Convert velocity from effective center to control inputs v, w
            v_new, w_new = lib.revert_v(pose, agent.new_velocity_.x, agent.new_velocity_.y)

            # Use if v, w are consistently exceeding bounds
            # v_new = saturate(v_new, MIN_LINEAR_V, MAX_LINEAR_V)
            # w_new = saturate(w_new, MIN_ANGULAR_V, MAX_ANGULAR_V)
            self.velocity_control(robot_id, v_new, w_new)

        return v_new, w_new

    # Visual servoing for object pickup
    def visual_servoing(self, robot_id, target_pos, pose):
        # vector from robot to object
        r = norm((target_pos[0] - pose[0], target_pos[1] - pose[1])) - self.object_distance_offset

        # orientation of robot w.r.t. line of sight
        delta = pose[2] - np.arctan2(target_pos[1] - pose[1], target_pos[0] - pose[0])
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta <= -np.pi:
            delta += 2 * np.pi

        # orientation of target w.r.t. line of sight
        theta = 0

        k = curvature(norm(r), theta, delta)
        v = compute_v(k, self.max_linear_velocity)
        w = k * v

        self.velocity_control(robot_id, linear_velocity=v, rotational_velocity=w)

        return r, delta, v, w

    # measures the objects in the same 1m^2 cell as the robot (with some noise)
    def measure(self, robot_id, objects, r=1.0, noise=None, sigma=0.1):
        pose, v = self.get_robot_state(robot_id)
        x, y = map(floor, pose[0:2])
        count = 0
        for obj in objects:
            if objects[obj] not in ("RECOVERED", "RETRIEVED"):
                u, v = self.get_object_state(obj)
                if x <= u <= x + r and y <= v <= y + r:
                    count += 1

        return round(np.random.default_rng().normal(count, sigma), 1) if noise == "GAUSSIAN" else count
