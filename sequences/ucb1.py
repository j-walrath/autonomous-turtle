import math
import logging
from utils.control import RobotControl
from utils import fsm
from utils import simulator_library as lib
import numpy as np

N = 100
M = 5


def in_cell(target, state):
    return target[0] <= state[0] <= target[0]+1 and target[1] <= state[1] <= target[1]+1


# SINGLE AGENT UCB TEST
def ucb1(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Single Robot UCB1...')

    mu = [3, 2]
    sig = [[0.4, 0], [0, 0.4]]
    var = 0.1
    field = np.zeros((M, M))
    visits = np.zeros((M, M))
    reward = np.zeros((M, M))
    T = N
    delta = 1

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
            lib.cycle_robot(pb, robot_fsm)
            lib.step(pb, 100)
            reward[i][j] = controller.measure(robot, robot_fsm.obj_states, noise="GAUSSIAN", sigma=var)

    # RUN UCB1
    for i in range(1, T+1):
        exp_mean = np.divide(reward, visits)
        Q = exp_mean + np.divide(math.sqrt(2 * math.log(i)), np.sqrt(visits))
        target = np.unravel_index(np.argmax(Q), Q.shape)

        robot_fsm.set_destination(lib.get_cell_coordinates(target[0], target[1]))
        lib.cycle_robot(pb, robot_fsm)

        lib.step(pb, 100)
        measurement = controller.measure(robot, robot_fsm.obj_states, noise="GAUSSIAN", sigma=var)
        reward[target] += measurement

        # PICKUP OBJECT
        for obj in objects:
            if object_states[obj] == "ON_GROUND" and in_cell(target, controller.get_object_state(obj)):
                robot_fsm.set_target(obj)
                break

        lib.cycle_robot(pb, robot_fsm)
        visits[target] += 1

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
