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

import numpy as np

import pybullet as p
import pybullet_utils.bullet_client as pbc
import pybullet_data

from fsm import RobotStateMachine

# DIRECTORIES & LOCATIONS
baseDir = './urdf_models/'
outDir = './outputs/'
planeModel = "./urdf_models/plane_with_dumpsters.urdf"
objModel = "./urdf_models/objects/object.urdf"
robotModel = "./urdf_models/tb_openmanipulator/trash_collect_robot_four_wheel.urdf"

# TODO: Track 'global' variables here (e.g. obj & robot ids)
DEFAULT_BOUNDS = (-5, 5)


def init_sim(numObjects, numRobots):  # PYBULLET INIT
    pb = pbc.BulletClient(connection_mode=p.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.807)
    pb.resetDebugVisualizerCamera(6, 0, -89, [0, 0, 0])  # overhead camera perspective

    # LOAD PLANE
    pb.loadURDF(planeModel, basePosition=[0, 0, 0], globalScaling=1.0)
    pb.changeDynamics(0, 0, lateralFriction=1.0, spinningFriction=0.0, rollingFriction=0.0)

    # LOAD OBJECTS
    objects = load_objects(pb, DEFAULT_BOUNDS[0], DEFAULT_BOUNDS[1], numObjects)
    object_states = {}
    for obj in objects:
        pos, orn = pb.getBasePositionAndOrientation(bodyUniqueId=obj)
        object_states[obj] = [pos, "ON_GROUND"]

    # LOAD ROBOTS
    robots = load_robots(pb, DEFAULT_BOUNDS[0], DEFAULT_BOUNDS[1], numRobots)
    fsm = {}
    for robot in robots:
        fsm[robot] = RobotStateMachine(pb, robot, max_linear_v=1.0, max_rotational_v=2 * np.pi)

    return pb, robots, fsm, objects, object_states


def load_objects(pb, lBound, uBound, numObjects):  # LOAD OBJECTS
    objects = []
    lower = lBound + 0.5
    upper = uBound - 0.5
    rng = np.random.default_rng()
    coords = (upper - lower) * rng.random(size=(numObjects, 2)) + lower
    for i in range(numObjects):
        objects.append(pb.loadURDF(objModel, basePosition=[coords[i, 0], coords[i, 1], 0.3], globalScaling=1.0))

    return objects


def load_robots(pb, lBound, uBound, numRobots):  # LOAD ROBOT(S)
    # TODO: Fix loading in multiple robots
    robots = []
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
            pb.changeDynamics(robots[-1], -1, maxJointVelocity=300)

    return robots


def step(pb, t):
    for _ in range(t):
        pb.stepSimulation()
        time.sleep(1./240.)

# RUN SIM
# TODO: Robots follow UCB algorithm
# TODO: Add live stats to the GUI
# TODO: Record frames/stats and save to output
if __name__ == "__main__":
    pb, robots, fsm, objects, object_states = init_sim()

    step(pb, 10000)

    pb.disconnect()
