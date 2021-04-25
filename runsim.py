"""Simple Simulator

A sparse simulator that loads the field, some objects, and some robots.

Currently used to test PyBullet features and capabilities. Does not
utilize many (if any) of the classes in ./utils, but probably replicates
(to some degree) their functionality.

By default, renders in GUI mode over direct. Hopefully saves video to ./outputs

Many values are hardcoded, but will be parameterized in the future once I
get a better sense of which values are variable.
"""

# IMPORTS
import logging

import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as pbc

import utils.simulator_library as lib
from utils.fsm import RobotStateMachine

from sequences.ucb1 import ucb1
from sequences.ucb1_multi import ucb1_multi
from sequences.function_test import function_test
from sequences.function_test import function_test_multi
from sequences.ray_test import ray_test
from sequences.visit_cells import visit_cells
from sequences.velocity_test import velocity_test

logging.basicConfig(level=logging.NOTSET)

# DIRECTORIES & LOCATIONS
baseDir = './urdf_models/'
outDir = './outputs/'

# CONSTANTS & GLOBAL VARIABLES
DEFAULT_BOUNDS = (-5, 5)   # area bounds for sim world
SIM_SEQUENCE = ucb1_multi

objects = []               # list of body unique object IDs
object_states = {}         # key: object ID, val: string state of object

robots = []                # list of body unique robot IDs
robot_fsms = {}            # key: robot ID, val: robot FSM instance


def generate_obj_coordinates(bounds, n):  # LOAD OBJECTS
    lower = bounds[0] + 0.5
    upper = bounds[1] - 0.5
    rng = np.random.default_rng()
    coords = (upper - lower) * rng.random(size=(n, 2)) + lower
    return coords


def generate_robot_coordinates(bounds, n):  # LOAD ROBOT(S)
    # TODO: Fix loading in multiple robots
    coords = []
    if n == 1:
        coords.append([0, 0])
    elif n > 1:
        for i in range(n):
            coords.append([bounds[0] + i, bounds[0]])
    else:
        return None

    return coords


def init_sim(numObjects=0, numRobots=0):  # PYBULLET INIT
    physClient = pbc.BulletClient(connection_mode=p.GUI)
    physClient.setAdditionalSearchPath(pybullet_data.getDataPath())
    physClient.setGravity(0, 0, -9.807)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # physClient.resetDebugVisualizerCamera(6, 0, -89, [0, 0, 0])  # overhead camera perspective
    physClient.resetDebugVisualizerCamera(4, 0, -89, [2.5, 2.5, 0])  # overhead camera perspective (pos grid)

    # LOAD PLANE
    lib.load_plane(physClient)

    # LOAD OBJECTS
    if numObjects:
        for obj in lib.load_objects(physClient, generate_obj_coordinates(DEFAULT_BOUNDS, numObjects)):
            objects.append(obj)

    # LOAD ROBOTS
    if numRobots:
        for robot in lib.load_robots(physClient, generate_robot_coordinates(DEFAULT_BOUNDS, numRobots)):
            robots.append(robot)

    return physClient


def init_states(physClient):
    for obj in objects:
        object_states[obj] = 'ON_GROUND'

    for robot in robots:
        robot_fsms[robot] = RobotStateMachine(physClient, object_states, robot_fsms, robot)


# RUN SIM
# TODO: Add live stats to the GUI
# TODO: Record frames/stats and save to output
if __name__ == "__main__":
    logging.info('Initializing GUI Simulator...')
    pb = init_sim(numObjects=0, numRobots=0)

    lib.step(pb, 100)

    logging.info('Initializing Object & Robot States...')
    init_states(pb)
    lib.step(pb, 100)

    logging.info('Running Simulation...')
    SIM_SEQUENCE(pb, objects, object_states, robots, robot_fsms)

    logging.info('Simulation Runtime Complete.')

    logging.info('Disconnecting Simulation...')
    pb.disconnect()
