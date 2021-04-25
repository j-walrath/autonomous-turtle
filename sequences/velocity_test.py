import logging
import time
import numpy as np

from utils.control import RobotControl
import utils.simulator_library as lib


def velocity_test(pb, objects, object_states, robots, robot_fsms):
    controller = RobotControl(pb)
    pb.resetDebugVisualizerCamera(2, 0, -89, [0, 0, 0])

    robot = lib.load_robots(pb, [[0, 0]])[0]

    for _ in range(10000):
        controller.velocity_control(robot, linear_velocity=3, rotational_velocity=1.0)
        for _ in range(int(lib.SIM_FREQUENCY/lib.CONTROL_FREQUENCY)):
            pb.stepSimulation()
            time.sleep(1. / lib.SIM_FREQUENCY)
        logging.debug('{} is moving at {:.2}m/s, {:.2} rad/s'.format(lib.NAMES[robot-1],
                                                                     np.linalg.norm(controller.get_robot_state(robot)[1][0:2]),
                                                                     controller.get_robot_state(robot)[1][2]))

    logging.debug('Returning to main...')
    return
