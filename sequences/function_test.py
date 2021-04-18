import time
import logging
from utils.control import RobotControl
from utils import fsm

CONTROL_FREQUENCY = 40
RADIUS = 0.5


def wait(pb, t):
    for _ in range(t):
        pb.stepSimulation()
        time.sleep(1. / 240.)


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


def function_test(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Single Robot Functions Test...')

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

                m = controller.measure(robot, robot_fsm.obj_states, r=RADIUS, noise="GAUSSIAN")
                actual = m_actual(controller, robot, objects, object_states)
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
