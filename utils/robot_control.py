import numpy as np
from numpy.linalg import norm as euclidean_norm
from scipy.linalg import logm

import pybullet as pb

DEG_TO_RAD = np.pi/180


def euler_to_rot_mat(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])


def normalize_vector(v):
    return v/np.linalg.norm(v)


def compute_rotational_velocity(yaw_robot, yaw_target):
    relative_orientation = yaw_target - yaw_robot
    
    relative_orientation_mat = np.array([[np.cos(relative_orientation), -np.sin(relative_orientation)],
                                          [np.sin(relative_orientation), np.cos(relative_orientation)]])

    return logm(relative_orientation_mat)[1, 0]


class RobotControl:
    minimum_linear_velocity = .1
    critical_distance = .5
    object_distance_offset = 0.16428811071136287
    object_distance_offset = 0.25
    
    def __init__(self, pb_client, max_linear_velocity=.2, max_rotational_velocity=5.0):
        self.pb_client = pb_client
        self.max_linear_velocity = max_linear_velocity
        self.max_rotational_velocity = max_rotational_velocity
        
    def get_manipulator_state(self, robot_id):
        joint_angles, velocity, _, _ = zip(*self.pb_client.getJointStates(bodyUniqueId=robot_id, jointIndices=range(11-4, 15-4)))

        return (joint_angles, velocity)

    def get_robot_state(self, robot_id):
        position, orientation = self.pb_client.getBasePositionAndOrientation(bodyUniqueId=robot_id)
        _, _, yaw = pb.getEulerFromQuaternion(orientation)
        pose = (position[0], position[1], pb.getEulerFromQuaternion(orientation)[2])

        linear_velocity, angular_velocity = self.pb_client.getBaseVelocity(bodyUniqueId=robot_id)
        velocity = (linear_velocity[0], linear_velocity[1], angular_velocity[2])

        return (pose, velocity)
    
    def gripper_control(self, robot_id, gripper_target_state):
        for i, value in enumerate(gripper_target_state):
            self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=15+i, controlMode=pb.POSITION_CONTROL, targetPosition=value)

    def manipulator_control(self, robot_id, arm_target_state):
        for i, value in enumerate(arm_target_state):
            self.pb_client.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=11+i-4, controlMode=pb.POSITION_CONTROL, targetPosition=value, maxVelocity=5.0*2)

    def gripper_pinch(self, robot_id):
        gripper_target_state = (-0.016812667692005963, -0.01000778853017392)
        
        self.gripper_control(robot_id, gripper_target_state)

    def gripper_release(self, robot_id):
        gripper_target_state = (0.01786004943591301, 0.021918541425825912)

        self.gripper_control(robot_id, gripper_target_state)

    def manipulator_origin(self, robot_id):
        arm_target_state = (DEG_TO_RAD*0, DEG_TO_RAD*-80, DEG_TO_RAD*60, DEG_TO_RAD*30)

        self.manipulator_control(robot_id, arm_target_state)

    def manipulator_lower(self, robot_id):
        arm_target_state = (-0.0016962233862355199, 1.2404879177129509, -0.901944524850455, 1.1624811955078364)

        self.manipulator_control(robot_id, arm_target_state)

    def manipulator_place(self, robot_id):
        arm_target_state = (DEG_TO_RAD*180-.2*np.random.random(), 0, .4, 1.1)

        self.manipulator_control(robot_id, arm_target_state)

    def velocity_control(self, robot_id, linear_velocity, rotational_velocity):
        left_wheel_velocity = linear_velocity/0.033 - rotational_velocity*0.027135999999999997/(.144*.033)
        right_wheel_velocity = linear_velocity/0.033 + rotational_velocity*0.027135999999999997/(.144*.033)

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
        
        distances_to_points = [euclidean_norm((point[5][0]-point[6][0], point[5][1]-point[6][1])) 
                               for point in closest_points if point[3] not in [0,1] and point[4] not in [0,1]]
        
        if len(distances_to_points) == 0: return []
        closest_point_index = np.argmin(distances_to_points)
        
        return closest_points[closest_point_index][6]
        
    def pose_control_with_collision_avoidance(self, robot_id, destination, avoidance_direction, weight_magnitude, weight_degree, threshold_distance=2.0):
        robot_pose, robot_velocity = self.get_robot_state(robot_id=robot_id)

        vector_to_destination = (destination[0]-robot_pose[0],
                                 destination[1]-robot_pose[1])
        unit_vector_to_destination = normalize_vector(vector_to_destination)
        unit_vector_yaw = np.array((np.cos(robot_pose[2]), np.sin(robot_pose[2])))

        correlation_coefficient = np.dot(unit_vector_yaw, unit_vector_to_destination)
        distance_to_destination = euclidean_norm(vector_to_destination)
        
        linear_velocity = min(self.max_linear_velocity, correlation_coefficient*distance_to_destination)
        linear_velocity = max(self.minimum_linear_velocity, linear_velocity)

        orientation_to_destination = np.arctan2(destination[1]-robot_pose[1], destination[0]-robot_pose[0])
        rotational_velocity = compute_rotational_velocity(robot_pose[2], orientation_to_destination)

        for robot2_id in avoidance_direction.keys():
            if robot2_id == robot_id: continue

            ## find closest point of robot [robot2_id] from robot [robot_id]
            closest_point_on_robot2 = self.get_closest_point(robot_id, robot2_id, robot_pose, threshold_distance)
            if len(closest_point_on_robot2) == 0: continue

            robot2_pose, robot2_velocity = self.get_robot_state(robot_id=robot2_id)
            robot2_velocity_magnitude = euclidean_norm(robot2_velocity)

            vector_to_robot2 = np.array((closest_point_on_robot2[0]-robot_pose[0], closest_point_on_robot2[1]-robot_pose[1]))
            distance_to_robot2 = euclidean_norm(vector_to_robot2)
            unit_vector_to_robot2 = normalize_vector(vector_to_robot2)
            
            correlation_coefficient = np.dot(unit_vector_yaw, unit_vector_to_robot2)

            if (distance_to_robot2 >= threshold_distance or correlation_coefficient <= 0): continue

            ## define obstacle avoidance rule
            if (avoidance_direction[robot2_id] == 0):
                rotation_direction = np.sign(np.cross((unit_vector_to_robot2[0], unit_vector_to_robot2[1], 0), 
                                                      (unit_vector_yaw[0], unit_vector_yaw[1], 0))[2])
                
            else:
                rotation_direction = avoidance_direction[robot2_id]

            target_orientation = np.arctan2(unit_vector_to_robot2[1], unit_vector_to_robot2[0]) + rotation_direction*weight_degree(distance_to_robot2)*np.pi/2*1.2
            rotational_velocity += weight_magnitude(distance_to_robot2)*compute_rotational_velocity(robot_pose[2], target_orientation)

        rotational_velocity = min(self.max_rotational_velocity, rotational_velocity)
        rotational_velocity = max(-self.max_rotational_velocity, rotational_velocity)
        
        self.velocity_control(robot_id, linear_velocity, rotational_velocity)

        return linear_velocity, rotational_velocity

    def pose_control_oa_v1(self, robot_id, pose_dest, pose_robot, laser_scan_data=[], gain_oa=2.0, dist_threshold=.8):
        vec_to_dest = (pose_dest[0]-pose_robot[0],
                           pose_dest[1]-pose_robot[1])
        unit_vec_to_dest = normalize_vector(vec_to_dest)
        unit_yaw_vec = np.array((np.cos(pose_robot[2]), np.sin(pose_robot[2])))

        cor_coeff = np.dot(unit_yaw_vec, unit_vec_to_dest)
        dist_to_dest = np.linalg.norm(vec_to_dest)
        linear_vel = min(self.max_linear_velocity, 1.0*cor_coeff*dist_to_dest)
        linear_vel = max(0.05, linear_vel)

        yaw_target = np.arctan2(pose_dest[1]-pose_robot[1], pose_dest[0]-pose_robot[0])
        rot_vec = compute_rotational_velocity(pose_robot[2], yaw_target)

        ## process laser scan data
        for data in laser_scan_data:
            x_b, y_b, _ = data

            vec_to_data_point = np.array((x_b-pose_robot[0], y_b-pose_robot[1]))
            dist_to_data_point = np.linalg.norm(vec_to_data_point)
            unit_vec_to_data_point = normalize_vector(vec_to_data_point)

            cor_coeff = np.dot(unit_yaw_vec, unit_vec_to_data_point)
            if dist_to_data_point >= dist_threshold or cor_coeff <= 0: continue

            if dist_to_data_point <= .4: linear_vel = 0.02

            weight = np.exp(-gain_oa*(dist_to_data_point-dist_threshold))
            rot_direction = np.sign(np.cross((unit_vec_to_data_point[0], unit_vec_to_data_point[1], 0), (unit_yaw_vec[0], unit_yaw_vec[1], 0))[2])
            yaw_target = np.arctan2(unit_vec_to_data_point[1], unit_vec_to_data_point[0]) + rot_direction*np.pi/2
            rot_vec += weight*compute_rotational_velocity(pose_robot[2], yaw_target)
            
        rot_vel = min(self.max_rotational_velocity, 1.0*rot_vec)
        rot_vel = max(-self.max_rotational_velocity, rot_vel)
    
        self.velocity_control(robot_id, linear_vel, rot_vel)

        return linear_vel, rot_vel

    ## Visual servoing for object pickup
    def visual_servoing(self, robot_id, target_position, pose):
        distance_to_target = np.linalg.norm((target_position[0]-pose[0], target_position[1]-pose[1])) - self.object_distance_offset
        orientation_to_target = np.arctan2(target_position[1]-pose[1], target_position[0]-pose[0]) - pose[2]
        
        rotational_velocity = min(self.max_rotational_velocity, compute_rotational_velocity(pose[2], np.arctan2(target_position[1]-pose[1], target_position[0]-pose[0])))
        rotational_velocity = max(-self.max_rotational_velocity, rotational_velocity)

        if np.abs(rotational_velocity) < 0.2:
            linear_velocity = min(self.max_linear_velocity, 1.0*distance_to_target)
            linear_velocity = max(-self.max_linear_velocity, linear_velocity)

        else:
            linear_velocity = 0

        self.velocity_control(robot_id, linear_velocity, rotational_velocity)

        return (np.abs(distance_to_target), np.abs(orientation_to_target), linear_velocity, rotational_velocity)
