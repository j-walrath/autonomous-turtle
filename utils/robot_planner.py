import itertools

import numpy as np
from numpy.random import choice as random_choice
from numpy.linalg import norm as euclidean_norm
from scipy.special import softmax

from rtree import index


class RobotPlanner:
    cell_width = .1
    cell_height = .1

    noise_level = .05
    revision_rate = 900
    regularization_weight = 1.0
    model_weight = 10.0

    def __init__(self,
                 robot_id=0,
                 number_of_robots = [10],
                 cluster_bounds=[[(-5, -5), (5, 5)]],
                 robot_population_state_limit=[20]):
        self.robot_id = robot_id
        self.number_of_clusters = len(cluster_bounds)
        self.robot_population_state_limit = robot_population_state_limit
        
        ## initialize cluster belief state and model parameters
        self.clusters_belief_state = {}
        self.model_parameters = {}
        for cluster_index in range(self.number_of_clusters):
            self.clusters_belief_state[cluster_index] = {}
            self.clusters_belief_state[cluster_index]["robot_population_state"] = number_of_robots[cluster_index]
            self.clusters_belief_state[cluster_index]["cells_belief_state"] = []
            self.clusters_belief_state[cluster_index]["cells_center"] = []
            self.clusters_belief_state[cluster_index]["cells_rtree"] = index.Index()

            x_min, y_min = cluster_bounds[cluster_index][0]
            x_max, y_max = cluster_bounds[cluster_index][1]
            x_range = np.arange(x_min-5, x_max+5, self.cell_width)
            y_range = np.arange(y_min-5, y_max+5, self.cell_height)
            for cell_index, (x, y) in enumerate(itertools.product(x_range, y_range)):
                self.clusters_belief_state[cluster_index]["cells_belief_state"] += [0]
                self.clusters_belief_state[cluster_index]["cells_center"] += [(x+self.cell_width/2, y+self.cell_height/2)]
                self.clusters_belief_state[cluster_index]["cells_rtree"].insert(cell_index, (x, y, x+self.cell_width, y+self.cell_height))

            self.model_parameters[cluster_index] = {}
            self.model_parameters[cluster_index]["a"] = 0.0
            self.model_parameters[cluster_index]["b"] = 0.0
            self.model_parameters[cluster_index]["object_relocation_rate"] = 0.0

    def update_clusters_belief_state(self, world_state=None, robot_population_state=None):
        if world_state != None:
            for cluster_index in self.clusters_belief_state.keys():
                self.clusters_belief_state[cluster_index]["cells_belief_state"] = np.zeros_like(self.clusters_belief_state[cluster_index]["cells_belief_state"])

                for object_index in world_state[cluster_index]["objects"].keys():
                    location, status = world_state[cluster_index]["objects"][object_index]

                    if status == "ON_GROUND":
                        for cell in self.clusters_belief_state[cluster_index]["cells_rtree"].intersection((location[0], location[1], location[0], location[1]), objects=True):
                            self.clusters_belief_state[cluster_index]["cells_belief_state"][cell.id] = 1.0

        if robot_population_state != None:
            for cluster_index in self.clusters_belief_state.keys():
                self.clusters_belief_state[cluster_index]["robot_population_state"] = robot_population_state[cluster_index]

    def assign_destination(self, robot_cluster_index):
        cells_belief_state = np.array(self.clusters_belief_state[robot_cluster_index]["cells_belief_state"])
        number_of_cells = cells_belief_state.shape[0]

        payoff = cells_belief_state + self.regularization_weight*cells_belief_state*(1-cells_belief_state)
        payoff_softmax = softmax(payoff/self.noise_level)

        cell_index = random_choice(a=range(number_of_cells), p=payoff_softmax)
        cell_center = self.clusters_belief_state[robot_cluster_index]["cells_center"][cell_index]

        return cell_center

    def revise_cluster_membership(self, robot_cluster_index):
        mean_belief_state = [np.sum(self.clusters_belief_state[cluster_index]["cells_belief_state"]) for cluster_index in self.clusters_belief_state.keys()]
        variance_belief_state = [np.sum(self.clusters_belief_state[cluster_index]["cells_belief_state"]*(1-self.clusters_belief_state[cluster_index]["cells_belief_state"]))
                                 for cluster_index in self.clusters_belief_state.keys()]
        payoff = np.array(mean_belief_state) + self.regularization_weight*np.array(variance_belief_state)
        smith_protocol = [np.maximum(p-payoff[robot_cluster_index], 0)/self.revision_rate for p in payoff]

        for cluster_index in self.clusters_belief_state.keys():
            if (self.clusters_belief_state[cluster_index]["robot_population_state"]>=self.robot_population_state_limit[cluster_index]):
                smith_protocol[cluster_index] = 0

        smith_protocol[robot_cluster_index] = 1-np.sum(smith_protocol)
        new_robot_cluster_index = random_choice(a=range(self.number_of_clusters), p=smith_protocol)

        return new_robot_cluster_index

    def revise_cluster_membership_model_based(self, robot_cluster_index):
        mean_belief_state = [np.sum(self.clusters_belief_state[cluster_index]["cells_belief_state"]) for cluster_index in self.clusters_belief_state.keys()]
        variance_belief_state = [np.sum(self.clusters_belief_state[cluster_index]["cells_belief_state"]*(1-self.clusters_belief_state[cluster_index]["cells_belief_state"]))
                                 for cluster_index in self.clusters_belief_state.keys()]
        payoff = np.array(mean_belief_state) + self.regularization_weight*np.array(variance_belief_state)

        for cluster_index in range(self.number_of_clusters):
            a = self.model_parameters[cluster_index]["a"]
            b = self.model_parameters[cluster_index]["b"]
            object_relocation_rate = self.model_parameters[cluster_index]["object_relocation_rate"]
            
            payoff[cluster_index] += self.model_weight*(a*self.clusters_belief_state[cluster_index]["robot_population_state"]**b + object_relocation_rate) # robot_population_state is the number of robots

        smith_protocol = [np.maximum(p-payoff[robot_cluster_index], 0)/self.revision_rate for p in payoff]

        for cluster_index in self.clusters_belief_state.keys():
            if (self.clusters_belief_state[cluster_index]["robot_population_state"]>=self.robot_population_state_limit[cluster_index]):
                smith_protocol[cluster_index] = 0

        smith_protocol[robot_cluster_index] = 1-np.sum(smith_protocol)
        new_robot_cluster_index = random_choice(a=range(self.number_of_clusters), p=smith_protocol)

        return new_robot_cluster_index

    def update_model_parameters(self, cluster_index, model_parameters):
        self.model_parameters[cluster_index] = model_parameters

    def get_clusters_belief_state(self):
        return self.clusters_belief_state
