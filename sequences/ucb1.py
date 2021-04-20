import math
import logging
from utils.control import RobotControl
from utils import fsm
from utils import simulator_library as lib
import numpy as np

N = 10
M = 5


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

    for obj in lib.load_objects(pb, object_locations):
        objects.append(obj)
        object_states[obj] = "ON_GROUND"

    controller = RobotControl(pb)

    robot = robots[0]
    robot_fsm: fsm.RobotStateMachine = robot_fsms[robot]

    logging.debug("Executing Simulation...")

    # INITIALIZE AGENT
    for i in range(M):
        for j in (range(M - 1, -1, -1) if i & 1 else range(M)):
            visits[i][j] += 1
            robot_fsm.set_destination(lib.get_cell_coordinates(i, j))

            while True:
                manipulator_state = controller.get_manipulator_state(robot)
                robot_state = controller.get_robot_state(robot)
                robot_fsm.run_once((manipulator_state, robot_state))
                lib.step(pb, int(lib.SIM_FREQUENCY/lib.CONTROL_FREQUENCY))

                if robot_fsm.current_state == "NONE":
                    lib.step(pb, 100)
                    reward[i][j] = controller.measure(robot, robot_fsm.obj_states, noise="GAUSSIAN", sigma=var)
                    break

    # RUN UCB1
    for i in range(1, T+1):
        exp_mean = np.divide(reward, visits)
        Q = exp_mean + np.divide(math.sqrt(2 * math.log(i)), np.sqrt(visits))
        target = np.unravel_index(np.argmax(Q), Q.shape)

        robot_fsm.set_destination(lib.get_cell_coordinates(target[0], target[1]))
        while True:
            manipulator_state = controller.get_manipulator_state(robot)
            robot_state = controller.get_robot_state(robot)
            robot_fsm.run_once((manipulator_state, robot_state))
            lib.step(pb, int(lib.SIM_FREQUENCY / lib.CONTROL_FREQUENCY))

            if robot_fsm.current_state == "NONE":
                lib.step(pb, 100)
                measurement = controller.measure(robot, robot_fsm.obj_states, noise="GAUSSIAN", sigma=var)
                reward[target] += measurement * (1 - delta**visits[target])
                visits[target] += 1

                robot_fsm.set_destination((0, 0))
                while True:
                    manipulator_state = controller.get_manipulator_state(robot)
                    robot_state = controller.get_robot_state(robot)
                    robot_fsm.run_once((manipulator_state, robot_state))
                    lib.step(pb, int(lib.SIM_FREQUENCY / lib.CONTROL_FREQUENCY))

                    if robot_fsm.current_state == "NONE":
                        break
                break
        logging.info("Time step: {:<2} | Target: {}".format(i, target))
        logging.info(visits)

    lib.step(pb, 100)

    # FINAL OUTPUT
    logging.info("Simulation Complete!")
    visits -= 1
    logging.info(visits)
    logging.info(field - visits)

    logging.debug("Returning to Main...")

    return
