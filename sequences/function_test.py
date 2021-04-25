import time
import logging

import utils.control
from utils.control import RobotControl
from utils.fsm import RobotStateMachine
import utils.simulator_library as lib

CONTROL_FREQUENCY = 40
RADIUS = 1


def function_test(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Single Robot Functions Test...')

    controller = RobotControl(pb)

    robot = robots[0]
    logging.debug('Robot ID: %s', robot)

    robot_fsm: RobotStateMachine = robot_fsms[robot]

    logging.debug("Executing Simulation...")

    cells = [(4, 4), (4, -4), (-4, -4), (-4, 4)]
    targets = [(3.5, 3.5), (3.5, -3.5), (-3.5, -3.5), (-3.5, 3.5)]
    coords = [(-3.8, -3.8), (-3.8, -3.6), (-3.8, -3.4), (-3.6, -3.8), (-3.4, -3.8),
              (-3.8, 3.8), (-3.8, 3.6), (-3.8, 3.4), (-3.6, 3.8), (-3.4, 3.8),
              (3.8, 3.8), (3.8, 3.6), (3.8, 3.4), (3.6, 3.8), (3.4, 3.8),
              (3.8, -3.8), (3.8, -3.6), (3.8, -3.4), (3.6, -3.8), (3.4, -3.8)]

    for obj in lib.load_objects(pb, coords):
        objects.append(obj)
        object_states[obj] = "ON_GROUND"

    # MEASURE CELLS
    for i in range(len(cells)):
        robot_fsm.set_destination(targets[i])
        logging.debug("Moving to Cell {}...".format(cells[i]))

        lib.cycle_robot(pb, robot_fsm)
        lib.step(pb, 100)

        m = controller.measure(robot, robot_fsm.obj_states, r=RADIUS, noise="GAUSSIAN")
        actual = controller.measure(robot, robot_fsm.obj_states, r=RADIUS)
        logging.debug("Robot measured {} objects (Actual = {}).".format(m, actual))

        lib.step(pb, 100)

        robot_fsm.set_destination((0, 0))
        logging.debug("Returning to Origin...")

        lib.cycle_robot(pb, robot_fsm)

    # RETRIEVE ONE OBJECT FROM EACH MEASURED CELL
    targets = [4, 9, 14, 19]
    for target in targets:
        robot_fsm.set_target(target)

        logging.debug("Retrieving object at {}...".format(controller.get_object_state(target)))

        lib.cycle_robot(pb, robot_fsm)

        # Only needed if utils.control.MAX_VOLUME is not 1
        robot_fsm.set_destination((0, 0))
        logging.debug("Returning to Origin...")
        lib.cycle_robot(pb, robot_fsm)

    lib.step(pb, 100)

    logging.debug("Returning to Main...")

    return


def function_test_multi(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Multi-Robot Functions Test...')

    controller = RobotControl(pb, robot_fsms)

    robot_coords = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    for robot in lib.load_robots(pb, robot_coords):
        robots.append(robot)
        robot_fsms[robot] = RobotStateMachine(pb, object_states, robot_fsms, robot,
                                              max_linear_v=utils.control.MAX_LINEAR_V)

    logging.debug('Loaded robots...')

    coords = [(-3.8, -3.8), (-3.8, -3.6), (-3.8, -3.4), (-3.6, -3.8), (-3.4, -3.8),
              (-3.8, 3.8), (-3.8, 3.6), (-3.8, 3.4), (-3.6, 3.8), (-3.4, 3.8),
              (3.8, 3.8), (3.8, 3.6), (3.8, 3.4), (3.6, 3.8), (3.4, 3.8),
              (3.8, -3.8), (3.8, -3.6), (3.8, -3.4), (3.6, -3.8), (3.4, -3.8)]
    for obj in lib.load_objects(pb, coords):
        objects.append(obj)
        object_states[obj] = "ON_GROUND"

    logging.debug('Loaded objects...')

    logging.debug("Executing Simulation...")

    cells = [(4, 4), (4, -4), (-4, -4), (-4, 4)]
    targets = [(3.5, 3.5), (3.5, -3.5), (-3.5, -3.5), (-3.5, 3.5)]
    # ASSIGN TARGETS
    for i in range(len(cells)):
        robot_fsms[robots[i]].set_destination((targets[i]))
        logging.debug("Moving Robot {} to Cell {}...".format(robots[i], cells[i]))

    # MEASURE CELLS
    lib.cycle_robots(pb, robot_fsms)
    for robot in robots:
        robot_fsm = robot_fsms[robot]
        m = controller.measure(robot, robot_fsm.obj_states, r=RADIUS, noise="GAUSSIAN")
        actual = controller.measure(robot, robot_fsm.obj_states, r=RADIUS)
        logging.debug("Robot {} measured {} objects (Actual = {}).".format(robot, m, actual))

        robot_fsm.set_destination(robot_coords[robot - 1])
        logging.debug("Robot {} returning to starting location...".format(robot))

    # RETURN TO START
    lib.cycle_robots(pb, robot_fsms)

    # RETRIEVE OBJECTS
    targets = [7, 12, 17, 22]
    for i in range(len(robots)):
        robot_fsms[robots[i]].set_target(targets[i])
        logging.debug("Robot {} retrieving object {}...".format(robots[i], targets[i]))

    lib.cycle_robots(pb, robot_fsms)

    # RETURN TO START
    for robot in robots:
        robot_fsms[robot].set_destination(robot_coords[robot - 1])
        logging.debug("Robot {} returning to starting location...".format(robot))
    lib.cycle_robots(pb, robot_fsms)

    lib.step(pb, 100)

    logging.debug("Returning to Main...")

    return
