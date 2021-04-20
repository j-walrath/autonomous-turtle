import time
import logging
from math import floor
from utils.control import RobotControl
from utils import fsm
import utils.simulator_library as lib

CONTROL_FREQUENCY = 40
RADIUS = 1.0


def measure_corners(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Single Robot Measurement Test...')

    controller = RobotControl(pb)

    cells = [(4, 4), (4, -4), (-4, -4), (-4, 4)]
    targets = [(3.5, 3.5), (3.5, -3.5), (-3.5, -3.5), (-3.5, 3.5)]
    coords = [(-3.8, -3.8), (-3.8, -3.6), (-3.8, -3.4), (-3.6, -3.8), (-3.4, -3.8),
              (-3.8, 3.8), (-3.8, 3.6), (-3.8, 3.4), (-3.6, 3.8), (-3.4, 3.8),
              (3.8, 3.8), (3.8, 3.6), (3.8, 3.4), (3.6, 3.8), (3.4, 3.8),
              (3.8, -3.8), (3.8, -3.6), (3.8, -3.4), (3.6, -3.8), (3.4, -3.8)]

    for obj in lib.load_objects(pb, coords):
        objects.append(obj)
        object_states[obj] = "ON_GROUND"

    robot = robots[0]
    logging.debug('Robot ID: %s', robot)

    robot_fsm: fsm.RobotStateMachine = robot_fsms[robot]

    logging.debug("Executing Simulation...")

    for i in range(len(cells)):
        robot_fsm.set_destination(targets[i])
        logging.debug("Moving to Cell {}...".format(cells[i]))

        while True:
            manipulator_state = controller.get_manipulator_state(robot)
            robot_state = controller.get_robot_state(robot)

            robot_fsm.run_once((manipulator_state, robot_state))

            for _ in range(int(240 / CONTROL_FREQUENCY)):
                pb.stepSimulation()
                time.sleep(1. / 240.)

            if robot_fsm.current_state == "NONE":
                for _ in range(100):
                    pb.stepSimulation()
                    time.sleep(1. / 240.)

                m = controller.measure(robot, robot_fsm.obj_states, r=RADIUS, noise="GAUSSIAN")
                actual = controller.measure(robot, robot_fsm.obj_states, r=RADIUS)
                logging.debug("Robot measured {} objects (Actual = {}).".format(m, actual))

                for _ in range(100):
                    pb.stepSimulation()
                    time.sleep(1. / 240.)

                robot_fsm.set_destination((0, 0))
                logging.debug("Returning to Origin...".format(cells[i]))
                while True:
                    manipulator_state = controller.get_manipulator_state(robot)
                    robot_state = controller.get_robot_state(robot)

                    robot_fsm.run_once((manipulator_state, robot_state))

                    for _ in range(int(240 / CONTROL_FREQUENCY)):
                        pb.stepSimulation()
                        time.sleep(1. / 240.)

                    if robot_fsm.current_state == "NONE":
                        break

                break

    for _ in range(100):
        pb.stepSimulation()
        time.sleep(1. / 240.)

    logging.debug("Returning to Main...")

    return
