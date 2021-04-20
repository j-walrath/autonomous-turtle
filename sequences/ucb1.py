import time
import logging
from utils.control import RobotControl
from utils import fsm
import numpy as np

CONTROL_FREQUENCY = 40
N = 10
M = 5


def wait(pb, t):
    for _ in range(t):
        pb.stepSimulation()
        time.sleep(1. / 240.)


# SINGLE AGENT UCB TEST
def ucb1(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Single Robot UCB1...')

    mu = [3, 2]
    sig = [[0.1, 0], [0, 0.1]]
    var = 0.1
    field = np.zeros((M, M))
    visits = np.zeros((M, M))
    reward = np.zeros((M, M))
    T = 10
    delta = 0.2

    object_locations = np.random.multivariate_normal(mu, sig, N)


    controller = RobotControl(pb)

    robot = robots[0]
    logging.debug('Robot ID: %s', robot)

    robot_fsm: fsm.RobotStateMachine = robot_fsms[robot]

    logging.debug("Executing Simulation...")

    cells = [(4, 4), (4, -4), (-4, -4), (-4, 4)]
    targets = [(3.5, 3.5), (3.5, -3.5), (-3.5, -3.5), (-3.5, 3.5)]

    # MEASURE CELLS
    for i in range(len(cells)):
        robot_fsm.set_destination(targets[i])
        logging.debug("Moving to Cell {}...".format(cells[i]))

        while True:
            manipulator_state = controller.get_manipulator_state(robot)
            robot_state = controller.get_robot_state(robot)

            robot_fsm.run_once((manipulator_state, robot_state))

            wait(pb, int(240 / CONTROL_FREQUENCY))

            if robot_fsm.current_state == "NONE":
                wait(pb, 100)

                m = controller.measure(robot, robot_fsm.obj_states, noise="GAUSSIAN")
                actual = controller.measure(robot, robot_fsm.obj_states,)
                logging.debug("Robot measured {} objects (Actual = {}).".format(m, actual))

                wait(pb, 100)

                robot_fsm.set_destination((0, 0))
                logging.debug("Returning to Origin...")
                while True:
                    manipulator_state = controller.get_manipulator_state(robot)
                    robot_state = controller.get_robot_state(robot)

                    robot_fsm.run_once((manipulator_state, robot_state))

                    wait(pb, int(240 / CONTROL_FREQUENCY))

                    if robot_fsm.current_state == "NONE":
                        break
                break

    # RETRIEVE ONE OBJECT FROM EACH MEASURED CELL
    targets = [4, 9, 14, 19]
    for target in targets:
        robot_fsm.set_target(target)

        logging.debug("Retrieving object at {}...".format(controller.get_object_state(target)))

        while True:
            manipulator_state = controller.get_manipulator_state(robot)
            robot_state = controller.get_robot_state(robot)

            robot_fsm.run_once((manipulator_state, robot_state))

            wait(pb, int(240 / CONTROL_FREQUENCY))

            if robot_fsm.current_state == "NONE":

                # Only needed if utils.control.MAX_VOLUME is not 1
                # robot_fsm.set_destination((0, 0))
                # logging.debug("Returning to Origin...")
                # while True:
                #     manipulator_state = controller.get_manipulator_state(robot)
                #     robot_state = controller.get_robot_state(robot)
                #
                #     robot_fsm.run_once((manipulator_state, robot_state))
                #
                #     wait(pb, int(240 / CONTROL_FREQUENCY))
                #
                #     if robot_fsm.current_state == "NONE":
                #         break
                break

    wait(pb, 100)

    logging.debug("Returning to Main...")

    return
