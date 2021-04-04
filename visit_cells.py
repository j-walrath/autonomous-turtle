import time
import logging
from control import RobotControl
import fsm

CONTROL_FREQUENCY = 40


def test1(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Single Robot Movement Test...')

    controller = RobotControl(pb)

    robot = robots[0]
    logging.debug('Robot ID: %s', robot)

    robot_fsm: fsm.RobotStateMachine = robot_fsms[robot]

    logging.debug("Executing Simulation...")

    for i in range(-5, 6):
        for j in range(-5, 6):
            robot_fsm.set_destination((i, j))
            logging.debug("Moving to Cell ({}, {})...".format(i, j))

            while True:
                manipulator_state = controller.get_manipulator_state(robot)
                robot_state = controller.get_robot_state(robot)

                robot_fsm.run_once(manipulator_state, robot_state)

                for _ in range(int(240 / CONTROL_FREQUENCY)):
                    pb.stepSimulation()
                    time.sleep(1. / 240.)

                if robot_fsm.current_state == "NONE":
                    break

    for _ in range(480):
        pb.stepSimulation()
        time.sleep(1. / 240.)

    logging.debug("Returning to Main...")

    return
