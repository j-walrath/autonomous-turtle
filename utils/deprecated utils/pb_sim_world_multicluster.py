import itertools
import numpy as np
from numpy.linalg import norm as euclidean_norm
import pybullet as pb

class PBSimWorld:
    dumpster_pos = (100, 100)

    plane_filename = "./urdf_models/plane_with_dumpsters.urdf"
    robot_model_filename = "./urdf_models/tb_openmanipulator/trash_collect_robot_four_wheel.urdf"
    object_model_filename = "./urdf_models/objects/object.urdf"

    def __init__(self,
                 pb_client,
                 number_of_robots = [10],
                 number_of_objects = [100],
                 cluster_bounds = [[(-5, -5), (5, 5)]],
                 collection_sites = [[(0,0), 5]],
                 model_scale=1.0):

        self.pb_client = pb_client
        self.number_of_robots = number_of_robots
        self.number_of_objects = number_of_objects
        self.number_of_clusters = len(cluster_bounds)
        self.cluster_bounds = cluster_bounds
        self.collection_sites = collection_sites
        self.model_scale = model_scale

        self.initialize_simulator()
    
    def initialize_simulator(self):
        self.pb_client.resetSimulation()

        ## load plane
        self.pb_client.loadURDF(fileName=self.plane_filename, basePosition=[0, 0, 0], globalScaling=1.0)
        self.pb_client.changeDynamics(0, 0, lateralFriction=1.0, spinningFriction=0.0, rollingFriction=0.0)

        self.robot_ids = []
        self.object_ids = []
        uniform_rv = np.random.default_rng().uniform

        robot_cluster_membership = {}
        for cluster_index, bound in enumerate(self.cluster_bounds):
            number_of_robots_cluster = self.number_of_robots[cluster_index]
            number_of_objects_cluster = self.number_of_objects[cluster_index]

            x_min, y_min = bound[0]
            x_max, y_max = bound[1]

            flag = False
            while flag != True:
                x_random = uniform_rv(low=x_min+.5, high=x_max-.5, size=number_of_robots_cluster)
                y_random = uniform_rv(low=y_min+.5, high=y_max-.5, size=number_of_robots_cluster)
                yaw_random = uniform_rv(low=-np.pi, high=np.pi, size=number_of_robots_cluster)
                
                flag = True
                for i,j in itertools.combinations(range(number_of_robots_cluster), 2):
                    distance_btw_points = euclidean_norm((x_random[i]-x_random[j],
                                                          y_random[i]-y_random[j]))

                    if distance_btw_points < 1.0:
                        flag = False
                        break
            
            ## load trash collection robots (turtlebots with open manipulator)
            for robot_index in range(number_of_robots_cluster):
                x = x_random[robot_index]
                y = y_random[robot_index]
                yaw = yaw_random[robot_index]
                quaternion_yaw = pb.getQuaternionFromEuler([0, 0, yaw])

                robot_id = self.pb_client.loadURDF(self.robot_model_filename,
                                                   basePosition=[x, y, 0.01], 
                                                   baseOrientation=quaternion_yaw,
                                                   globalScaling=self.model_scale)
                self.robot_ids.append(robot_id)
                robot_cluster_membership[robot_id] = cluster_index
            
                self.pb_client.changeDynamics(robot_id, -1, maxJointVelocity=300)

            x_random = uniform_rv(low=x_min+.5, high=x_max-.5, size=number_of_objects_cluster)
            y_random = uniform_rv(low=y_min+.5, high=y_max-.5, size=number_of_objects_cluster)
            
            ## load objects
            for object_index in range(number_of_objects_cluster):
                x = x_random[object_index]
                y = y_random[object_index]

                object_id = self.pb_client.loadURDF(self.object_model_filename, 
                                                    basePosition=(x, y, 0.3), 
                                                    globalScaling=self.model_scale)
                self.object_ids.append(object_id)
                   
        self.pb_client.setGravity(0, 0, -10)
        
        return robot_cluster_membership

    def initialize_states(self):
        world_state = {}
        for cluster_index in range(-1, self.number_of_clusters):
            world_state[cluster_index]= {}
            world_state[cluster_index]["objects"] = {}
        
        for object_id in self.object_ids:
            position, orientation = self.pb_client.getBasePositionAndOrientation(bodyUniqueId=object_id)
            object_cluster_index = self.get_cluster_index(position)

            ## check if the object is on the ground
            n_contact_with_plane = len(self.pb_client.getContactPoints(bodyA=0, bodyB=object_id, linkIndexA=-1))
            n_contact_with_dumpster = len(self.pb_client.getContactPoints(bodyA=1, bodyB=object_id, linkIndexA=0))

            if n_contact_with_plane > 0:
                world_state[object_cluster_index]["objects"][object_id] = [position, "ON_GROUND"]

            else:
                world_state[object_cluster_index]["objects"][object_id] = [position, "REMOVED"]
                
        self.world_state = world_state        

    ## update world state
    def update_states(self):
        for cluster_index in range(-1, self.number_of_clusters):
            for object_id in list(self.world_state[cluster_index]["objects"].keys()):
                position, orientation = self.pb_client.getBasePositionAndOrientation(bodyUniqueId=object_id)
                object_position = (position[0], position[1])
                object_cluster_index = self.get_cluster_index(position)

                n_contact_with_plane = len(self.pb_client.getContactPoints(bodyA=0, bodyB=object_id, linkIndexA=-1))
                n_contact_with_dumpster = len(self.pb_client.getContactPoints(bodyA=1, bodyB=object_id, linkIndexA=0))

                if n_contact_with_dumpster > 0:
                    self.world_state[cluster_index]["objects"][object_id] = [position, "REMOVED"]
                    
                elif n_contact_with_plane > 0:
                    if self.world_state[cluster_index]["objects"][object_id][1] == "ASSIGNED":
                        self.world_state[cluster_index]["objects"][object_id][0] = position

                    else:
                        if object_cluster_index != cluster_index:
                            self.world_state[cluster_index]["objects"].pop(object_id)
                            self.world_state[object_cluster_index]["objects"][object_id] = [position, "ON_GROUND"]
                            
                        else:
                            self.world_state[cluster_index]["objects"][object_id] = [position, "ON_GROUND"]
                            
                else:
                    self.world_state[cluster_index]["objects"][object_id][0] = position

                ## put objects nearby dumpster if they are in one of collection sites
                for site_location, radius in self.collection_sites:
                    if euclidean_norm(np.array(object_position)-np.array(site_location)) <= radius:
                        dx = 1.0*(np.random.random()-.5)
                        dy = 1.0*(np.random.random()-.5)
                        dz = 1.0*(np.random.random()-.5)

                        self.pb_client.resetBasePositionAndOrientation(object_id, 
                                                                       posObj=(self.dumpster_pos[0]+dx, self.dumpster_pos[1]+dy, .5+dz), 
                                                                       ornObj=(0,0,0,1))

                        self.world_state[cluster_index]["objects"].pop(object_id)
                        self.world_state[-1]["objects"][object_id] = [(self.dumpster_pos[0]+dx, self.dumpster_pos[1]+dy, .5+dz), "REMOVED"]
                    
                        break
        
    def relocate_object_from_dumpster(self, cluster_index):
        uniform_rv = np.random.default_rng().uniform
        
        bound = self.cluster_bounds[cluster_index]        
        x_min, y_min = bound[0]
        x_max, y_max = bound[1]

        x_random = uniform_rv(low=x_min+.5, high=x_max-.5, size=1)
        y_random = uniform_rv(low=y_min+.5, high=y_max-.5, size=1)
        
        if len(self.world_state[-1]["objects"].keys()) > 0:
            object_id = list(self.world_state[-1]["objects"].keys())[0]
            
            self.pb_client.resetBasePositionAndOrientation(object_id, 
                                                           posObj=(x_random[0], y_random[0], .3), 
                                                           ornObj=(0,0,0,1))

            self.world_state[-1]["objects"].pop(object_id)
        
        else:
            object_id = self.pb_client.loadURDF(self.object_model_filename, 
                                                basePosition=(x_random[0], y_random[0], .3),
                                                globalScaling=self.model_scale)
            self.object_ids.append(object_id)
        
        self.world_state[cluster_index]["objects"][object_id] = [(x_random[0], y_random[0], .3), "ON_GROUND"]
                                            
    ## find the index of the cluster closest to pos
    def get_cluster_index(self, pos):
        distances_to_clusters = []
        
        for idx, bound in enumerate(self.cluster_bounds):
            x_min, y_min = bound[0]
            x_max, y_max = bound[1]
            
            distance = 0
            if pos[0]>=x_min and pos[0]<x_max and pos[1]>=y_min and pos[1]<y_max:
                distance = 0
                
            else:
                dx = 0
                if pos[0]<x_min: dx = x_min-pos[0]
                elif pos[0]>=x_max: dx = pos[0]-x_max
                    
                dy = 0
                if pos[1]<y_min: dy = y_min-pos[1]
                elif pos[1]>=y_max: dy = pos[1]-y_max
                    
                distance = np.sqrt(dx**2+dy**2)
                
            distances_to_clusters.append(distance)
                
        cluster_index = np.argmin(distances_to_clusters)
                
        return cluster_index
        
    def get_robot_ids(self):
        return self.robot_ids

    def get_object_ids(self):
        return self.object_ids

    def get_world_state(self):
        return self.world_state
