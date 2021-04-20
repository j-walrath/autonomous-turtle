import time
import logging
from utils.control import RobotControl
from utils import fsm
import utils.simulator_library as lib

CONTROL_FREQUENCY = 40
RADIUS = 1


def function_test(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Single Robot Functions Test...')

    controller = RobotControl(pb)

    robot = robots[0]
    logging.debug('Robot ID: %s', robot)

    robot_fsm: fsm.RobotStateMachine = robot_fsms[robot]

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

        while True:
            lib.cycle_robot(robot_fsm)
            lib.step(pb, int(240 / CONTROL_FREQUENCY))

            if robot_fsm.current_state == "NONE":
                lib.step(pb, 100)

                m = controller.measure(robot, robot_fsm.obj_states, r=RADIUS, noise="GAUSSIAN")
                actual = controller.measure(robot, robot_fsm.obj_states, r=RADIUS)
                logging.debug("Robot measured {} objects (Actual = {}).".format(m, actual))

                lib.step(pb, 100)

                robot_fsm.set_destination((0, 0))
                logging.debug("Returning to Origin...")
                while True:
                    lib.cycle_robot(robot_fsm)

                    lib.step(pb, int(240 / CONTROL_FREQUENCY))

                    if robot_fsm.current_state == "NONE":
                        break
                break

    # RETRIEVE ONE OBJECT FROM EACH MEASURED CELL
    targets = [4, 9, 14, 19]
    for target in targets:
        robot_fsm.set_target(target)

        logging.debug("Retrieving object at {}...".format(controller.get_object_state(target)))

        while True:
            lib.cycle_robot(robot_fsm)
            lib.step(pb, int(240 / CONTROL_FREQUENCY))

            if robot_fsm.current_state == "NONE":
                # Only needed if utils.control.MAX_VOLUME is not 1
                # robot_fsm.set_destination((0, 0))
                # logging.debug("Returning to Origin...")
                # while True:
                #     lib.cycle_robot(robot_fsm)
                #     lib.step(pb, int(240 / CONTROL_FREQUENCY))
                #
                #     if robot_fsm.current_state == "NONE":
                #         break
                break

    lib.step(pb, 100)

    logging.debug("Returning to Main...")

    return
