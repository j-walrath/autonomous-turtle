import numpy as np
from numpy.linalg import norm
from scipy.linalg import logm
from math import floor
import logging

import pybullet as pb

DEG_TO_RAD = np.pi / 180


def euler_to_rot_mat(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw)],
                     [np.sin(yaw), np.cos(yaw)]])


def normalize_vector(v):
    return v / norm(v)


def compute_rotational_velocity(yaw_robot, yaw_target):
    relative_orientation = yaw_target - yaw_robot

    relative_orientation_mat = np.array([[np.cos(relative_orientation), -np.sin(relative_orientation)],
                                         [np.sin(relative_orientation), np.cos(relative_orientation)]])

    return 2*logm(relative_orientation_mat)[1, 0]


def curvature(r, theta, delta):
    # ratio of rate of change in theta to the rate of change in r
    k1 = 1

    # timescale factor between fast subsystem and slow manifold
    k2 = 10

    return -(1/r)*(k2*(delta-np.arctan(-k1*theta)) + (1 + k1/(1+(k1*theta)**2))*np.sin(delta))


def compute_v(k, vmax):
    # higher beta = velocity drops more quickly as a function of k
    beta = 0.4

    # higher gamma = sharper peak for v/vmax vs k curve
    gamma = 1

    return vmax / (1 + (beta*abs(k)**gamma))


class RobotControl:
    minimum_linear_velocity = 0
    critical_distance = .5
    activation_threshold = 15 * DEG_TO_RAD
    object_distance_offset = 0.16428811071136287
    object_distance_offset = 0.3

    def __init__(self, pb_client, max_linear_velocity=1, max_rotational_velocity=5.0):
        self.pb_client = pb_client
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

    def velocity_control(self, robot_id, linear_velocity, rotational_velocity):
        left_wheel_velocity = linear_velocity / 0.033 - rotational_velocity * 0.027135999999999997 / (.144 * .033)
        right_wheel_velocity = linear_velocity / 0.033 + rotational_velocity * 0.027135999999999997 / (.144 * .033)

        self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                             jointIndex=0,
                                             controlMode=pb.VELOCITY_CONTROL,
                                             targetVelocity=left_wheel_velocity,
                                             force=500,
                                             velocityGain=.1)
        self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                             jointIndex=1,
                                             controlMode=pb.VELOCITY_CONTROL,
                                             targetVelocity=right_wheel_velocity,
                                             force=500,
                                             velocityGain=.1)

        self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                             jointIndex=2,
                                             controlMode=pb.VELOCITY_CONTROL,
                                             targetVelocity=left_wheel_velocity,
                                             force=500,
                                             velocityGain=.1)
        self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id,
                                             jointIndex=3,
                                             controlMode=pb.VELOCITY_CONTROL,
                                             targetVelocity=right_wheel_velocity,
                                             force=500,
                                             velocityGain=.1)

    def get_closest_point(self, robot1_id, robot2_id, robot1_pose, distance):
        closest_points = self.pb_client.getClosestPoints(bodyA=robot1_id, bodyB=robot2_id, distance=distance)

        distances_to_points = [norm((point[5][0] - point[6][0], point[5][1] - point[6][1]))
                               for point in closest_points if point[3] not in [0, 1] and point[4] not in [0, 1]]

        if len(distances_to_points) == 0:
            return []

        closest_point_index = np.argmin(distances_to_points)
        return closest_points[closest_point_index][6]

    def pose_control(self, robot_id, destination):
        pose, v_current = self.get_robot_state(robot_id=robot_id)
        yaw = pose[2]

        # vector from robot to target
        r = np.array(destination) - np.array(pose[0:2])

        # orientation of target w.r.t. line of sight from robot to target (-pi, pi]
        theta = 0

        # orientation of robot w.r.t. line of sight (-pi, pi]
        delta = yaw - np.arctan2(r[1], r[0])
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta <= -np.pi:
            delta += 2 * np.pi

        k = curvature(norm(r), theta, delta)
        v = compute_v(k, self.max_linear_velocity)
        w = k * v

        self.velocity_control(robot_id, linear_velocity=v, rotational_velocity=w)

        return v, w

    # CURRENTLY DEPRECATED
    def pose_control_oa_v1(self, robot_id, pose_dest, pose_robot, laser_scan_data=[], gain_oa=2.0, dist_threshold=.8):
        vec_to_dest = (pose_dest[0] - pose_robot[0],
                       pose_dest[1] - pose_robot[1])
        unit_vec_to_dest = normalize_vector(vec_to_dest)
        unit_yaw_vec = np.array((np.cos(pose_robot[2]), np.sin(pose_robot[2])))

        cor_coeff = np.dot(unit_yaw_vec, unit_vec_to_dest)
        dist_to_dest = np.linalg.norm(vec_to_dest)
        linear_vel = min(self.max_linear_velocity, 1.0 * cor_coeff * dist_to_dest)
        linear_vel = max(0.05, linear_vel)

        yaw_target = np.arctan2(pose_dest[1] - pose_robot[1], pose_dest[0] - pose_robot[0])
        rot_vec = compute_rotational_velocity(pose_robot[2], yaw_target)

        # process laser scan data
        for data in laser_scan_data:
            x_b, y_b, _ = data

            vec_to_data_point = np.array((x_b - pose_robot[0], y_b - pose_robot[1]))
            dist_to_data_point = np.linalg.norm(vec_to_data_point)
            unit_vec_to_data_point = normalize_vector(vec_to_data_point)

            cor_coeff = np.dot(unit_yaw_vec, unit_vec_to_data_point)
            if dist_to_data_point >= dist_threshold or cor_coeff <= 0: continue

            if dist_to_data_point <= .4: linear_vel = 0.02

            weight = np.exp(-gain_oa * (dist_to_data_point - dist_threshold))
            rot_direction = np.sign(np.cross((unit_vec_to_data_point[0], unit_vec_to_data_point[1], 0),
                                             (unit_yaw_vec[0], unit_yaw_vec[1], 0))[2])
            yaw_target = np.arctan2(unit_vec_to_data_point[1], unit_vec_to_data_point[0]) + rot_direction * np.pi / 2
            rot_vec += weight * compute_rotational_velocity(pose_robot[2], yaw_target)

        rot_vel = min(self.max_rotational_velocity, 1.0 * rot_vec)
        rot_vel = max(-self.max_rotational_velocity, rot_vel)

        self.velocity_control(robot_id, linear_vel, rot_vel)

        return linear_vel, rot_vel

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
            u, v = self.get_object_state(obj)
            if x <= u <= x+r and y <= v <= y+r and objects[obj] not in ("RECOVERED", "RETRIEVED"):
                count += 1

        return int(np.random.default_rng().normal(count, sigma)) if noise == "GAUSSIAN" else count
