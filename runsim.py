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
import pybullet_data

# DIRECTORIES
baseDir = './urdf_models/'
outDir = './outputs/'

# PYBULLET INIT
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.807)
p.resetDebugVisualizerCamera(8, 0, -89, [0, 0, 0])  # overhead camera perspective (camera gets weird at -90 deg pitch)

# LOAD PLANE
p.loadURDF("./urdf_models/plane_with_dumpsters.urdf", basePosition=[0, 0, 0], globalScaling=1.0)
p.changeDynamics(0, 0, lateralFriction=1.0, spinningFriction=0.0, rollingFriction=0.0)

# LOAD OBJECTS
lBound = -5.0
uBound = 5.0
numObjects = 100
rng = np.random.default_rng()
coords = (uBound - lBound) * rng.random(size=(numObjects, 2)) + lBound
for i in range(numObjects):
    p.loadURDF("./urdf_models/objects/object.urdf", basePosition=[coords[i, 0], coords[i, 1], 0.3], globalScaling=1.0)

# LOAD ROBOT(S)
pos = [0, 0, 0.5]
orn = p.getQuaternionFromEuler([0, 0, 0])
p.loadURDF("./urdf_models/tb_openmanipulator/trash_collect_robot_four_wheel.urdf", pos, orn)

# RUN SIM
for _ in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)


p.disconnect()
