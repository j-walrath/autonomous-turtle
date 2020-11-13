import numpy as np
from numpy.random import choice as random_choice
from numpy.linalg import norm as euclidean_norm
import pybullet as pb

from .robot_control import RobotControl
from .sensor_models import SensorModels

DEG_TO_RAD = np.pi/180
MAX_TRASH_VOLUME = 10
OBJECT_STATUS = ["ON_GROUND", "ASSIGNED", "REMOVED"]

def normalize_vector(v):
    return v/np.linalg.norm(v)

class ManipulatorStateMachine:
    def __init__(self, pb_client, robot_id):
        self.pb_client = pb_client
        
        self.handlers = {"LOWER": self.lower, 
                         "GRAB": self.grab,
                         "PLACE": self.place,
                         "RELEASE": self.release,
                         "ORIGIN": self.origin}

        self.handler = self.origin
        self.current_state = "NONE"

        self.robot_id = robot_id
        self.object_id = None
        self.constraint_id = None
        self.trash_volume_collected = random_choice(range(MAX_TRASH_VOLUME))

        self.pb_robot_control = RobotControl(pb_client)
        self.pb_sensor_models = SensorModels(pb_client)
        
        self.arm_target_state = [0]*4

    def release(self, manipulator_state):
        self.pb_client.removeConstraint(self.constraint_id)
        
        self.object_id = None
        self.constraint_id = None
        
        return "ORIGIN"

    def grab(self, manipulator_state):        
        self.constraint_id = self.pb_client.createConstraint(parentBodyUniqueId=self.robot_id, 
                                                             parentLinkIndex=14-4, 
                                                             childBodyUniqueId=self.object_id, 
                                                             childLinkIndex=-1, 
                                                             jointType=pb.JOINT_FIXED, 
                                                             jointAxis=[1,0,0], 
                                                             parentFramePosition=[.075, 0, 0], 
                                                             childFramePosition=[0,0,0])
        
        return "PLACE"
    
    def lower(self, manipulator_state):
        arm_target_state = (-0.0016962233862355199, 1.2404879177129509, -0.901944524850455, 1.1624811955078364)

        if self.current_state == "LOWER": 
            if np.linalg.norm(np.array(arm_target_state)-np.array(manipulator_state[0][:4]))<.005*2: 
                return "GRAB"

        else:
            self.pb_robot_control.manipulator_control(self.robot_id, arm_target_state)

            self.current_state = "LOWER"

        return "NONE"

    def place(self, manipulator_state):
        if self.current_state == "PLACE": 
            if np.linalg.norm(np.array(self.arm_target_state)-np.array(manipulator_state[0][:4]))<.005*2:
                return "RELEASE"

        else:
            self.arm_target_state = (DEG_TO_RAD*180-.2*np.random.random(), 0, .4, 1.1)
            self.pb_robot_control.manipulator_control(self.robot_id, self.arm_target_state)

            self.current_state = "PLACE"

        return "NONE"

    def origin(self, manipulator_state):
        arm_target_state = (5.08753421763152e-07, -1.3962637001751304, 1.0471974880549502, 0.523599283823225)

        if self.current_state == "ORIGIN": 
            if np.linalg.norm(np.array(arm_target_state)-np.array(manipulator_state[0][:4]))<.01: 
                return "DONE"

        else:        
            self.pb_robot_control.manipulator_control(self.robot_id, arm_target_state)

            self.current_state = "ORIGIN" 

        return "NONE"

    def empty_basket(self):
        self.trash_volume_collected = 0

    def reinitialize(self):
        self.handler = self.lower
        self.current_state = "NONE"
        
    def run_once(self, manipulator_state):
        new_state = self.handler(manipulator_state)      

        if new_state is "DONE":
            self.trash_volume_collected += 1

            if self.trash_volume_collected >= MAX_TRASH_VOLUME:
                return "RETRIEVE"

            return "DONE"

        elif new_state is not "NONE":
            self.handler = self.handlers[new_state]

        return 'INPROCESS'

class RobotStateMachine:
    maximum_destination_timeout = 40*5
    maximum_escape_timeout = 40*2
    maximum_visual_servo_timeout = 40*5
    collection_sites = [((0,0), 5)] # location, range
    
    threshold_distance = 1.0
    gain_weight_magnitude = 5.0
    gain_weight_degree = 5.0
    
    def __init__(self,
                 pb_client,
                 robot_id,
                 cluster_index=0,
                 max_linear_velocity=.2,
                 max_rotational_velocity=5.0):
        self.pb_client = pb_client
        self.handlers = {"PICKUP": self.pickup, 
                         "MOVE": self.move,
                         "VISUALSERVO": self.visual_servo,
                         "RETRIEVE": self.retrieve}
        self.handler = self.move

        self.robot_id = robot_id
        self.destination = None
        self.cluster_index = cluster_index
        self.cluster_bound = None
        self.cluster_transitioning = False
        
        self.world_state = None
        self.object_to_pickup = None
        
        self.distance_to_destination = 0
        self.destination_timeout = 0
        self.escape_timeout = self.maximum_escape_timeout
        self.visual_servo_timeout = 0
        
        self.manipulator_fsm = ManipulatorStateMachine(pb_client, robot_id)

        self.max_linear_velocity = max_linear_velocity
        self.max_rotational_velocity = max_rotational_velocity
        
        self.pb_robot_control = RobotControl(pb_client,
                                             max_linear_velocity=max_linear_velocity,
                                             max_rotational_velocity=max_rotational_velocity)
        self.pb_sensor_models = SensorModels(pb_client)

    def pickup(self, state):
        manipulator_state = state[0]

        result = self.manipulator_fsm.run_once(manipulator_state)

        if result is "DONE":
            try:
                self.world_state["objects"][self.object_to_pickup][1] = "REMOVED"
            except KeyError:
                pass
                
            self.object_to_pickup = None

            return "MOVE"

        elif result is "RETRIEVE":
            try:
                self.world_state["objects"][self.object_to_pickup][1] = "REMOVED"
            except KeyError:
                pass
            
            self.object_to_pickup = None

            return "RETRIEVE"

        return "NONE"

    def move(self, state):
        robot_pose, robot_velocity = state[1]
        
        if self.cluster_transitioning:
            x_min, y_min = self.cluster_bound[0]
            x_max, y_max = self.cluster_bound[1]
            if (x_min-5<=robot_pose[0] and y_min-5<=robot_pose[1]) and (x_max+5>robot_pose[0] and y_max+5>robot_pose[1]):
                self.pb_robot_control.max_linear_velocity = self.max_linear_velocity
                self.cluster_transitioning = False
                
        if self.escape_timeout < self.maximum_escape_timeout: self.escape_timeout += 1
            
        ## if an object detected, transition to visual servoing
        if self.world_state != None and self.escape_timeout >= self.maximum_escape_timeout and self.cluster_transitioning == False:
            for object_id in self.world_state["objects"].keys():
                if self.world_state["objects"][object_id][1] != "ON_GROUND": continue

                object_position = self.world_state["objects"][object_id][0]
                vector_to_object = np.array((object_position[0]-robot_pose[0], object_position[1]-robot_pose[1]))
                vector_to_object_normized = normalize_vector(vector_to_object)
                
                number_of_rays = 10+1
                del_th = np.pi/(number_of_rays-1)
                ray_from = (robot_pose[0]+.3*vector_to_object_normized[0], 
                            robot_pose[1]+.3*vector_to_object_normized[1], .01)
                
                rays_to = []
                for ray_index in range(number_of_rays):
                    ray_to_x = ray_from[0] + object_position[0]*np.cos(ray_index*del_th-np.pi/2) - object_position[1]*np.sin(ray_index*del_th-np.pi/2)
                    ray_to_y = ray_from[1] + object_position[0]*np.sin(ray_index*del_th-np.pi/2) + object_position[1]*np.cos(ray_index*del_th-np.pi/2)
                    
                    rays_to.append((ray_to_x, ray_to_y, .01))
                    
                ray_test_result = self.pb_client.rayTestBatch(rayFromPositions=[ray_from]*number_of_rays,
                                                              rayToPositions=rays_to)
    
                for data in ray_test_result:
                    if data[0] != object_id: continue

                ## if the robot's current position is .5 meter from the object, transition to visual servoing
                if euclidean_norm(vector_to_object) <= .5:
                    self.pb_robot_control.velocity_control(self.robot_id, 0, 0)
                    self.object_to_pickup = object_id
                    self.world_state["objects"][object_id][1] = "ASSIGNED"
                    self.visual_servo_timeout = 0

                    return "VISUALSERVO"

        if self.destination != None:
            distance_to_destination = euclidean_norm((self.destination[0]-robot_pose[0], self.destination[1]-robot_pose[1]))

            ## increase the timeout counter when there is no improvement in reaching the destination
            if self.distance_to_destination <= distance_to_destination:
                self.destination_timeout += 1

                if self.destination_timeout >= self.maximum_destination_timeout:
                    self.pb_robot_control.velocity_control(self.robot_id, 0, 0)
                    self.destination = None
                    
                    return "NONE"

            self.distance_to_destination = distance_to_destination
            
            if self.distance_to_destination > .3:
                weight_magnitude = lambda distance: np.exp(self.gain_weight_magnitude*(self.threshold_distance-distance))
                weight_degree = lambda distance: 1/(1+np.exp(-self.gain_weight_degree*(self.threshold_distance-distance)))
                linear_vel, rot_vel = self.pb_robot_control.pose_control_with_collision_avoidance(self.robot_id, 
                                                                                                  self.destination, 
                                                                                                  self.avoidance_direction[self.robot_id], 
                                                                                                  weight_magnitude, 
                                                                                                  weight_degree, 
                                                                                                  threshold_distance=self.threshold_distance)
            
            else:
                self.pb_robot_control.velocity_control(self.robot_id, 0, 0)
                self.destination = None

        return "NONE"

    def visual_servo(self, state):
        robot_pose, robot_velocity = state[1]

        try:
            if self.world_state["objects"][self.object_to_pickup][1] == "REMOVED":
                self.object_to_pickup = None
                return "MOVE"
        
            object_position = self.world_state["objects"][self.object_to_pickup][0]
            
        except KeyError:
            self.object_to_pickup = None
            return "MOVE"

        del_distance, del_orientation, linear_velocity, rotational_velocity = self.pb_robot_control.visual_servoing(self.robot_id, object_position, robot_pose)

        ## stop the robot if there is an obstacle near the robot
        unit_yaw_vector = np.array((np.cos(robot_pose[2]), np.sin(robot_pose[2])))
        laser_scan_data = self.pb_sensor_models.laser_scans(robot_id=self.robot_id)
        for data in laser_scan_data:
            x_b, y_b, _ = data

            vector_to_data_point = np.array((x_b-robot_pose[0], y_b-robot_pose[1]))
            distance_to_data_point = euclidean_norm(vector_to_data_point)
            unit_vector_to_data_point = normalize_vector(vector_to_data_point)

            corrlation_coefficient = np.dot(unit_yaw_vector, unit_vector_to_data_point)
            if distance_to_data_point <= .4 and corrlation_coefficient >= 0: 
                self.pb_robot_control.velocity_control(self.robot_id, 0, 0)
                self.visual_servo_timeout += 1
                
                if self.visual_servo_timeout >= self.maximum_visual_servo_timeout:
                    if self.world_state["objects"][self.object_to_pickup][1] == "ASSIGNED":
                        self.world_state["objects"][self.object_to_pickup][1] = "ON_GROUND"

                    self.object_to_pickup = None
                    self.escape_timeout = 0
                    
                    return "MOVE"

                return "NONE"
        
        if del_distance <= .05 and del_orientation < 10*np.pi/180:
            self.pb_robot_control.velocity_control(self.robot_id, 0, 0)
            self.manipulator_fsm.reinitialize()
            self.manipulator_fsm.object_id = self.object_to_pickup
            
            return "PICKUP"

        return "NONE"

    def retrieve(self, state):
        robot_pose, robot_velocity = state[1]

        ## find a nearest collection site
        collection_site_index = np.argmin([euclidean_norm(np.array((robot_pose[0], robot_pose[1]))-collection_location) 
                                           for collection_location, collection_range in self.collection_sites])
        
        collection_location, collection_range = self.collection_sites[collection_site_index] ## for now, only use the first collection site
        distance_to_collection_site = euclidean_norm((collection_location[0]-robot_pose[0], collection_location[1]-robot_pose[1]))

        if distance_to_collection_site > .8*collection_range:            
            weight_magnitude = lambda distance: np.exp(self.gain_weight_magnitude*(self.threshold_distance-distance))
            weight_degree = lambda distance: 1/(1+np.exp(-self.gain_weight_degree*(self.threshold_distance-distance)))
            linear_velocity, rotational_velocity = self.pb_robot_control.pose_control_with_collision_avoidance(self.robot_id, 
                                                                                                               collection_location, 
                                                                                                               self.avoidance_direction[self.robot_id], 
                                                                                                               weight_magnitude, 
                                                                                                               weight_degree, 
                                                                                                               threshold_distance=self.threshold_distance)
            
        else:
            self.pb_robot_control.velocity_control(self.robot_id, 0, 0)
            self.manipulator_fsm.empty_basket()
            self.destination = None

            return "MOVE"

        return "NONE"

    def set_destination(self, destination):
        self.destination = destination
        self.destination_timeout = 0

    def update_cluster_membership(self, cluster_index, cluster_bound):
        self.cluster_index = cluster_index
        self.cluster_bound = cluster_bound
        self.cluster_transitioning = True
        self.destination = None
        
    def set_cluster_index(self, cluster_index):
        self.cluster_index = cluster_index
        self.destination = None

    def update_world_state(self, world_state):
        self.world_state = world_state

    def run_once(self, state):
        new_state = self.handler(state)

        if new_state is not "NONE":
            self.handler = self.handlers[new_state]
    
    def breakdown(self):
        self.pb_robot_control.velocity_control(self.robot_id, 0, 0)

        return
