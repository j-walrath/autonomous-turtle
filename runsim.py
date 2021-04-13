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
import time
import logging

import numpy as np

import pybullet as p
import pybullet_utils.bullet_client as pbc
import pybullet_data

from fsm import RobotStateMachine
from test1 import test1
from visit_cells import visit_cells
from corner_measurements import measure_corners

logging.basicConfig(level=logging.NOTSET)

# DIRECTORIES & LOCATIONS
baseDir = './urdf_models/'
outDir = './outputs/'
planeModel = "./urdf_models/plane_with_dumpsters.urdf"
objModel = "./urdf_models/objects/object.urdf"
robotModel = "./urdf_models/tb_openmanipulator/trash_collect_robot_four_wheel.urdf"

# CONSTANTS & GLOBAL VARIABLES
DEFAULT_BOUNDS = (-5, 5)   # area bounds for sim world

objects = []               # list of body unique object IDs
object_states = {}         # key: object ID, val: string state of object

robots = []                # list of body unique robot IDs
robot_fsms = {}            # key: robot ID, val: robot FSM instance


def init_sim(numObjects=10, numRobots=2):  # PYBULLET INIT
    pb = pbc.BulletClient(connection_mode=p.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.807)
    pb.resetDebugVisualizerCamera(6, 0, -89, [0, 0, 0])  # overhead camera perspective

    # LOAD PLANE
    pb.loadURDF(planeModel, basePosition=[0, 0, 0], globalScaling=1.0)
    pb.changeDynamics(0, -1, lateralFriction=3.0, spinningFriction=0.03, rollingFriction=0.03, restitution=0.5)

    # LOAD OBJECTS
    load_objects(pb, DEFAULT_BOUNDS[0], DEFAULT_BOUNDS[1], numObjects)

    # LOAD ROBOTS
    load_robots(pb, DEFAULT_BOUNDS[0], DEFAULT_BOUNDS[1], numRobots)

    return pb


def load_objects(pb, lBound, uBound, numObjects):  # LOAD OBJECTS
    lower = lBound + 0.5
    upper = uBound - 0.5
    rng = np.random.default_rng()
    coords = (upper - lower) * rng.random(size=(numObjects, 2)) + lower
    for i in range(numObjects):
        objects.append(pb.loadURDF(objModel, basePosition=[coords[i, 0], coords[i, 1], 0.3], globalScaling=1.0))
        pb.setCollisionFilterGroupMask(objects[-1], -1, 0, 0)


def load_robots(pb, lBound, uBound, numRobots):  # LOAD ROBOT(S)
    # TODO: Fix loading in multiple robots
    orn = p.getQuaternionFromEuler([0, 0, 0])

    if numRobots == 1:
        robots.append(p.loadURDF(robotModel, [0, 0, 0.5], orn))
    else:
        lower = lBound + 0.5
        upper = uBound - 0.5
        rng = np.random.default_rng()
        coords = (upper - lower) * rng.random(size=(numRobots, 2)) + lower

        for i in range(numRobots):
            robots.append(pb.loadURDF(robotModel, [coords[i, 0], coords[i, 1], 0.5], orn))
            pb.changeDynamics(robots[-1], -1, maxJointVelocity=300, lateralFriction=1.0, rollingFriction=0.03, restitution=0.7)


def init_states(pb):
    for obj in objects:
        object_states[obj] = 'ON_GROUND'

    for robot in robots:
        robot_fsms[robot] = RobotStateMachine(pb, object_states, robot, max_linear_v=5)


def step(pb, t):
    for _ in range(t):
        pb.stepSimulation()
        time.sleep(1. / 240.)


# RUN SIM
# TODO: Robots follow UCB algorithm
# TODO: Add live stats to the GUI
# TODO: Record frames/stats and save to output
if __name__ == "__main__":
    numObjects = 0
    numRobots = 1
    sequence = measure_corners

    logging.info('Initializing GUI Simulator...')
    pb = init_sim(numObjects, numRobots)

    coords = [(-3.8, -3.8), (-3.8, -3.6), (-3.8, -3.4), (-3.6, -3.8), (-3.4, -3.8),
              (-3.8, 3.8), (-3.8, 3.6), (-3.8, 3.4), (-3.6, 3.8), (-3.4, 3.8),
              (3.8, 3.8), (3.8, 3.6), (3.8, 3.4), (3.6, 3.8), (3.4, 3.8),
              (3.8, -3.8), (3.8, -3.6), (3.8, -3.4), (3.6, -3.8), (3.4, -3.8)]
    for c in coords:
        objects.append(pb.loadURDF(objModel, basePosition=[c[0], c[1], 0.3], globalScaling=1.0))
        # pb.setCollisionFilterGroupMask(objects[-1], -1, 0, 0)

    step(pb, 100)

    logging.info('Initializing Object & Robot States...')
    init_states(pb)
    step(pb, 100)

    logging.info('Running Simulation...')
    sequence(pb, objects, object_states, robots, robot_fsms)

    logging.info('Simulation Runtime Complete.')

    logging.info('Disconnecting Simulation...')
    pb.disconnect()
