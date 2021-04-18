import time
import logging
from utils.control import RobotControl
from utils import fsm

CONTROL_FREQUENCY = 40
RADIUS = 0.5


def m_actual(controller, robot, objects, object_states):
    count = 0
    pose, _ = controller.get_robot_state(robot)
    x, y = pose[0:2]

    for obj in objects:
        u, v = controller.get_object_state(obj)

        if x - RADIUS <= u <= x + RADIUS \
                and y - RADIUS <= v <= y + RADIUS \
                and object_states[obj] not in ("RECOVERED", "RETRIEVED"):
            count += 1

    return count


def measure_corners(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Single Robot Measurement Test...')

    controller = RobotControl(pb)

    robot = robots[0]
    logging.debug('Robot ID: %s', robot)

    robot_fsm: fsm.RobotStateMachine = robot_fsms[robot]

    logging.debug("Executing Simulation...")

    cells = [(4, 4), (4, -4), (-4, -4), (-4, 4)]
    targets = [(3.5, 3.5), (3.5, -3.5), (-3.5, -3.5), (-3.5, 3.5)]

    for i in range(len(cells)):
        robot_fsm.set_destination((targets[i][0], targets[i][1]))
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
                actual = m_actual(controller, robot, objects, object_states)
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
