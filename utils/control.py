import numpy as np
from numpy.linalg import norm
from scipy.linalg import logm
from math import floor
from utils.rvo.agent import Agent
import logging
from utils import simulator_library as lib

import pybullet as pb

DEG_TO_RAD = np.pi / 180
MAX_LINEAR_V = 3.5


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def normalize_vector(v):
    return v / norm(v)


def normalize_angle(theta):
    if theta > np.pi:
        theta -= 2 * np.pi
    elif theta <= -np.pi:
        theta += 2 * np.pi

    return theta


def rotate(vector, angle):
    return np.around(np.dot(rotation_matrix(angle), vector))


def curvature(r, theta, delta):
    # ratio of rate of change in theta to the rate of change in r
    k1 = 1.5

    # timescale factor between fast subsystem and slow manifold
    k2 = 8

    return -(1/r)*(k2*(delta-np.arctan(-k1*theta)) + (1 + k1/(1+(k1*theta)**2))*np.sin(delta))


def compute_v(k, vmax):
    # higher beta = velocity drops more quickly as a function of k
    beta = 0.6

    # higher gamma = sharper peak for v/vmax vs k curve
    gamma = 1

    return vmax / (1 + (beta*abs(k)**gamma))


def get_rays(pose, length, height):
    (x, y, yaw) = pose
    sideLength = length * 0.6
    factor = 1.5
    rayFrom = []
    rayTo = []

    rayFrom.append([x + 0.15 * np.cos(yaw), y + 0.15 * np.sin(yaw), height])
    rayTo.append([rayFrom[0][0] + length * np.cos(yaw), rayFrom[0][1] + length * np.sin(yaw), height])

    for i in np.linspace(0.02, 0.11, 6):
        dx = i / length * (rayFrom[0][1] - rayTo[0][1])
        dy = i / length * (rayFrom[0][0] - rayTo[0][0])
        rayFrom.append([rayFrom[0][0] + dx, rayFrom[0][1] + dy, height])
        # rayTo.append([rayTo[0][0] + dx, rayTo[0][1] + dy, height])
        rayTo.append(
            [rayFrom[-1][0] + length * np.cos(yaw - (factor * i)), rayFrom[-1][1] + length * np.sin(yaw - (factor * i)),
             height])

    for i in np.linspace(3*np.pi/4, 0, 15, endpoint=False):
        rayFrom.append(rayFrom[6])
        rayTo.append(
            [rayFrom[6][0] + sideLength * np.cos(yaw + i), rayFrom[6][1] + sideLength * np.sin(yaw + i), height])

    for i in np.linspace(-0.02, -0.11, 6):
        dx = i / length * (rayFrom[0][1] - rayTo[0][1])
        dy = i / length * (rayFrom[0][0] - rayTo[0][0])
        rayFrom.append([rayFrom[0][0] + dx, rayFrom[0][1] + dy, height])
        # rayTo.append([rayTo[0][0] + dx, rayTo[0][1] + dy, height])
        rayTo.append(
            [rayFrom[-1][0] + length * np.cos(yaw - (factor * i)), rayFrom[-1][1] + length * np.sin(yaw - (factor * i)),
             height])

    for i in np.linspace(3*np.pi/4, 0, 15, endpoint=False):
        rayFrom.append(rayFrom[-1])
        rayTo.append(
            [rayFrom[-1][0] + sideLength * np.cos(yaw + i), rayFrom[-1][1] + sideLength * np.sin(yaw + i), height])

    return rayFrom, rayTo


class RobotControl:
    minimum_linear_velocity = 0
    critical_distance = .5
    activation_threshold = 15 * DEG_TO_RAD
    # object_distance_offset = 0.16428811071136287
    object_distance_offset = 0.3

    def __init__(self, pb_client, max_linear_velocity=MAX_LINEAR_V, max_rotational_velocity=5.0):
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
        delta = normalize_angle(yaw - np.arctan2(r[1], r[0]))

        k = curvature(norm(r), theta, delta)
        v = compute_v(k, self.max_linear_velocity)
        w = k * v

        return v, w

    # CURRENTLY DEPRECATED
    def smart_pose_control(self, robot_id, destination):
        pose, v_current = self.get_robot_state(robot_id)
        yaw = pose[2]
        v, w = self.pose_control(robot_id, destination)

        rays_from, rays_to = get_rays(pose, 5.0, 0.055)
        ray_results = pb.rayTestBatch(rays_from, rays_to)

        neighbors = set()
        for (neighbor_id, _, _, _, _) in ray_results:
            if neighbor_id not in (-1, 0, robot_id) and pb.getNumJoints(neighbor_id) > 1:
                neighbors.add(neighbor_id)

        if not neighbors:
            self.velocity_control(robot_id, linear_velocity=v, rotational_velocity=w)
            return v, w
        else:
            logging.debug('{} is moving around neighbors with IDs: {}'.format(robot_id, neighbors))
            agent_neighbors = []
            for neighbor in neighbors:
                neighbor_pose, neighbor_v = self.get_robot_state(neighbor)
                agent_neighbors.append(lib.get_agent(neighbor, neighbor_pose[0:2], neighbor_v[0:2]))

            agent = lib.get_agent(robot_id, pose=pose[0:2], v=v_current[0:2])
            agent.insert_agent_neighbors(agent_neighbors)

            # Convert desired control inputs v, w into preferred velocity vector v_pref
            vec_v = np.array([v * np.cos(yaw), v * np.sin(yaw)])
            theta = w * 1 / lib.CONTROL_FREQUENCY   # theta = w * dt
            v_pref = rotate(vec_v, theta)

            # Get adjusted velocity from ORCA given v_pref
            agent.pref_velocity_ = lib.get_vec(v_pref)
            agent.compute_new_velocity()

            # Convert adjusted velocity into actual v, w control inputs
            v_adjusted = np.array([agent.new_velocity_.x, agent.new_velocity_.y])
            phi = normalize_angle(np.arctan2(v_adjusted[1], v_adjusted[0]))
            v_new = norm(rotate(v_adjusted, (phi - yaw)))
            w_new = (phi - yaw) * lib.CONTROL_FREQUENCY     # w = theta / dt

            self.velocity_control(robot_id, linear_velocity=v_new, rotational_velocity=w_new)

        return v, w

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
                if x <= u <= x+r and y <= v <= y+r:
                    count += 1

        return round(np.random.default_rng().normal(count, sigma), 1) if noise == "GAUSSIAN" else count
