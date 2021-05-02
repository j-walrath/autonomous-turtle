import logging
import time

import pybullet as p
import math
import utils.simulator_library as lib
from utils.control import RobotControl
import numpy as np


def ray_test(pb, objects, object_states, robots, robot_fsms):
    controller = RobotControl(pb, robot_fsms, max_linear_velocity=5)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    pb.resetDebugVisualizerCamera(2, 0, -89, [0, 0, 0])

    useGui = True
    x_, y_ = 0, 0
    robots = lib.load_robots(pb, [[x_, y_]])
    for _ in range(100):
        pb.stepSimulation()
        time.sleep(1./240.)
    # robots = lib.load_robots(pb, [[3, 3], [-5, -2]])
    # objects = lib.load_objects(pb, [[0.7, 0.56], [0.5, 0.68], [0.45, 0.24]])
    # logging.debug("Objects have {} joints.".format(pb.getNumJoints(objects[0])))

    rayFrom = []
    rayTo = []
    rayIds = []

    numRays = 51

    rayLen = 5

    rayHitColor = [1, 0, 0]
    rayMissColor = [0, 1, 0]

    replaceLines = True

    # for k in range(numRays):
    #     rayFrom.append([0, 0, 0.2])
    #     rayTo.append([
    #         rayLen * math.sin(2. * math.pi * float(k) / numRays),
    #         rayLen * math.cos(2. * math.pi * float(k) / numRays), 0.064
    #     ])
    #     if replaceLines:
    #         rayIds.append(p.addUserDebugLine(rayFrom[k], rayTo[k], rayMissColor))
    #     else:
    #         rayIds.append(-1)

    length = 3.0
    width = 3.0
    sideLength = length
    height = 0.015
    factor = 1.5

    (x, y, yaw), _ = controller.get_robot_state(robots[-1])
    rayFrom.append([x * np.cos(yaw), y * np.sin(yaw), height])
    rayTo.append([rayFrom[0][0] + length * np.cos(yaw), rayFrom[0][1] + length * np.sin(yaw), height])
    if replaceLines:
        rayIds.append(p.addUserDebugLine(rayFrom[0], rayTo[0], rayMissColor, lineWidth=width, parentObjectUniqueId=robots[-1]))
    else:
        rayIds.append(-1)

    for i in np.linspace(0, 2 * np.pi, 80):
        rayFrom.append(rayFrom[-1])
        rayTo.append([rayFrom[-1][0] + length * np.cos(yaw+i), rayFrom[-1][1] + length * np.sin(yaw+i), height])

        if replaceLines:
            rayIds.append(p.addUserDebugLine(rayFrom[-1], rayTo[-1], rayMissColor, lineWidth=width))
        else:
            rayIds.append(-1)

    # for i in np.linspace(np.pi/4, 2*np.pi - np.pi/4, 7):
    #     rayFrom.append(rayFrom[-1])
    #     rayTo.append([rayFrom[-1][0] + length * np.cos(yaw+i), rayFrom[-1][1] + length * np.sin(yaw+i), height])
    #
    #     if replaceLines:
    #         rayIds.append(p.addUserDebugLine(rayFrom[0], rayTo[0], rayMissColor))
    #     else:
    #         rayIds.append(-1)

    # for i in np.linspace(0.02, 0.11, 6):
    #     dx = i/length * (rayFrom[0][1]-rayTo[0][1])
    #     dy = i/length * (rayFrom[0][0]-rayTo[0][0])
    #     rayFrom.append([rayFrom[0][0] + dx, rayFrom[0][1] + dy, height])
    #     # rayTo.append([rayTo[0][0] + dx, rayTo[0][1] + dy, height])
    #     rayTo.append([rayFrom[-1][0] + length * np.cos(yaw-(factor*i)), rayFrom[-1][1] + length * np.sin(yaw-(factor*i)), height])
    #
    #     if replaceLines:
    #         rayIds.append(p.addUserDebugLine(rayFrom[0], rayTo[0], rayMissColor, lineWidth=width))
    #     else:
    #         rayIds.append(-1)
    #
    # for i in np.linspace(-0.02, -0.11, 6):
    #     dx = i / length * (rayFrom[0][1] - rayTo[0][1])
    #     dy = i / length * (rayFrom[0][0] - rayTo[0][0])
    #     rayFrom.append([rayFrom[0][0] + dx, rayFrom[0][1] + dy, height])
    #     # rayTo.append([rayTo[0][0] + dx, rayTo[0][1] + dy, height])
    #     rayTo.append([rayFrom[-1][0] + length * np.cos(yaw-(factor*i)), rayFrom[-1][1] + length * np.sin(yaw-(factor*i)), height])
    #
    #     if replaceLines:
    #         rayIds.append(p.addUserDebugLine(rayFrom[0], rayTo[0], rayMissColor, lineWidth=width))
    #     else:
    #         rayIds.append(-1)

    # for i in np.linspace(0.47*np.pi, np.pi/20, 15, endpoint=False):
    #     rayFrom.append(rayFrom[-1])
    #     rayTo.append([rayFrom[-1][0] + sideLength * np.cos(yaw+i), rayFrom[-1][1] + sideLength * np.sin(yaw+i), height])
    #
    #     if replaceLines:
    #         rayIds.append(p.addUserDebugLine(rayFrom[0], rayTo[0], rayMissColor, lineWidth=width))
    #     else:
    #         rayIds.append(-1)
    #
    # for i in np.linspace(-0.47*np.pi, -np.pi/20, 15, endpoint=False):
    #     rayFrom.append(rayFrom[6])
    #     rayTo.append([rayFrom[6][0] + sideLength * np.cos(yaw+i), rayFrom[6][1] + sideLength * np.sin(yaw+i), height])
    #
    #     if replaceLines:
    #         rayIds.append(p.addUserDebugLine(rayFrom[0], rayTo[0], rayMissColor, lineWidth=width))
    #     else:
    #         rayIds.append(-1)

    numSteps = 327680
    for i in range(numSteps):
        p.stepSimulation()
        for j in range(8):
            results = p.rayTestBatch(rayFrom, rayTo, j + 1)

        # for i in range (10):
        #	p.removeAllUserDebugItems()

        if (useGui):
            if (not replaceLines):
                p.removeAllUserDebugItems()

            for i in range(numRays):
                hitObjectUid = results[i][0]

                if (hitObjectUid < 0):
                    hitPosition = [0, 0, 0]
                    p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i], lineWidth=width)
                else:
                    hitPosition = results[i][3]
                    p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i], lineWidth=width)
