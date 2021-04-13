import time
import logging
from control import RobotControl
import fsm

CONTROL_FREQUENCY = 40


def visit_cells(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Single Robot Movement Test...')

    controller = RobotControl(pb)

    robot = robots[0]
    logging.debug('Robot ID: %s', robot)

    robot_fsm: fsm.RobotStateMachine = robot_fsms[robot]

    logging.debug("Executing Simulation...")

    # for i in range(-5, 6):
    #     r = range(-5, 6) if i & 1 else range(5, -6, -1)
    #     for j in r:
    #         robot_fsm.set_destination((i, j))
    #         logging.debug("Moving to Location ({}, {})...".format(i, j))
    #
    #         while True:
    #             manipulator_state = controller.get_manipulator_state(robot)
    #             robot_state = controller.get_robot_state(robot)
    #
    #             robot_fsm.run_once((manipulator_state, robot_state))
    #
    #             for _ in range(int(240 / CONTROL_FREQUENCY)):
    #                 pb.stepSimulation()
    #                 time.sleep(1. / 240.)
    #
    #             if robot_fsm.current_state == "NONE":
    #                 break

    targets = [(0, 2), (3, 2), (-3, -3), (0, 0), (2, -4), (1, 3), (-2, 2), (0, 0)]
    for target in targets:
        robot_fsm.set_destination(target)
        logging.debug("Moving to Location ({}, {})...".format(target[0], target[1]))

        while True:
            manipulator_state = controller.get_manipulator_state(robot)
            robot_state = controller.get_robot_state(robot)

            robot_fsm.run_once((manipulator_state, robot_state))

            for _ in range(int(240 / CONTROL_FREQUENCY)):
                pb.stepSimulation()
                time.sleep(1. / 240.)

            if robot_fsm.current_state == "NONE":
                for _ in range(240):
                    pb.stepSimulation()
                    time.sleep(1. / 240.)
                break

    # d = (-2, 2)
    # robot_fsm.set_destination(d)
    # logging.debug("Moving to Location {}...".format(d))

    while True:
        manipulator_state = controller.get_manipulator_state(robot)
        robot_state = controller.get_robot_state(robot)

        robot_fsm.run_once((manipulator_state, robot_state))

        for _ in range(int(240 / CONTROL_FREQUENCY)):
            pb.stepSimulation()
            time.sleep(1. / 240.)

        # logging.debug("FSM State: {}".format(robot_fsm.current_state))
        # if robot_fsm.destination is None:
        #     break

        if robot_fsm.current_state == "NONE":
            break

    for _ in range(480):
        pb.stepSimulation()
        time.sleep(1. / 240.)

    logging.debug("Returning to Main...")

    return
