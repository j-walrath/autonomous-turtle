import time
import logging
from control import RobotControl
import fsm

CONTROL_FREQUENCY = 40


def test1(pb, objects, object_states, robots, robot_fsms):
    logging.info('Running Single Robot/Single Object Test...')

    controller = RobotControl(pb)

    obj = objects[0]
    logging.debug('Object ID: %s', obj)

    robot = robots[0]
    logging.debug('Robot ID: %s', robot)

    robot_fsm: fsm.RobotStateMachine = robot_fsms[robot]

    obj_pos = controller.get_object_state(obj)
    robot_fsm.set_destination(obj_pos)
    logging.debug('Object Position: (%d, %d)', obj_pos[0], obj_pos[1])

    logging.debug("Executing Simulation...")
    while object_states[obj] != "REMOVED":
        manipulator_state = controller.get_manipulator_state(robot)
        robot_state = controller.get_robot_state(robot)

        robot_fsm.run_once((manipulator_state, robot_state))

        for _ in range(int(240 / CONTROL_FREQUENCY)):
            pb.stepSimulation()
            time.sleep(1. / 240.)

    logging.debug("Returning to Main...")

    return
