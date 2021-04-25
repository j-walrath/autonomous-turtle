import time
from typing import List, Dict
import pybullet as p

import utils.control
from utils.fsm import RobotStateMachine
from utils.control import RobotControl
from utils.rvo.agent import Agent
from utils.rvo.vector import Vector2

OBJ_MODEL = "./urdf_models/objects/object.urdf"
ROBOT_MODEL = "./urdf_models/tb_openmanipulator/trash_collect_robot_four_wheel.urdf"
PLANE_MODEL = "./urdf_models/plane_with_dumpsters.urdf"

SIM_FREQUENCY = 240
CONTROL_FREQUENCY = 40
NAMES = ["Fluffy", "Oogway", "Crush", "Franklin", "Genbu", "Yertle", "Leonardo", "Raphael", "Donatello", "Michelangelo"]


def load_plane(pb, position=None, lateralFriction=3.0, spinningFriction=0.03, rollingFriction=0.03, restitution=0.5,
               scaling=1.0):
    if position is None:
        position = [0, 0, 0]

    plane = pb.loadURDF(PLANE_MODEL, basePosition=position, globalScaling=scaling)
    pb.changeDynamics(0, -1, lateralFriction=lateralFriction, spinningFriction=spinningFriction,
                      rollingFriction=rollingFriction, restitution=restitution)

    return plane


def load_objects(pb, locations, collision=False):
    objects = []
    for loc in locations:
        objects.append(pb.loadURDF(OBJ_MODEL, basePosition=[loc[0], loc[1], 0.3], globalScaling=1.0,
                                   flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES))

        if not collision:
            # Do not collide with robots or other objects
            pb.setCollisionFilterGroupMask(objects[-1], -1, 0, 0)

            # Do collide with the ground plane
            pb.setCollisionFilterPair(objects[-1], 0, -1, -1, 1)

    return objects


def load_robots(pb, locations, collision=True):
    robots = []
    orn = pb.getQuaternionFromEuler([0, 0, 0])
    for loc in locations:
        robots.append(pb.loadURDF(ROBOT_MODEL, basePosition=[loc[0], loc[1], 0.5], baseOrientation=orn,
                                  flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES))
        pb.changeDynamics(robots[-1], -1, maxJointVelocity=300, lateralFriction=1.0, rollingFriction=0.03,
                          restitution=0.7)
        step(pb, 100)

        if not collision:
            # Do not collide with other robots
            pb.setCollisionFilterGroupMask(robots[-1], -1, 0, 0)

            # Do collide with the ground plane
            pb.setCollisionFilterPair(robots[-1], 0, -1, -1, 1)

    return robots


def cycle_robot(pb, fsm: RobotStateMachine, controller: RobotControl = None):
    if controller is None:
        controller = fsm.control

    while True:
        manipulator_state = controller.get_manipulator_state(fsm.robot)
        robot_state = controller.get_robot_state(fsm.robot)
        fsm.run_once((manipulator_state, robot_state))

        step(pb, int(SIM_FREQUENCY/CONTROL_FREQUENCY))

        if fsm.current_state == "NONE":
            break


def cycle_robots(pb, fsms: Dict[int, RobotStateMachine], controller: RobotControl = None):
    while True:
        for fsm in fsms.values():
            if controller is None:
                controller = fsm.control
            manipulator_state = controller.get_manipulator_state(fsm.robot)
            robot_state = controller.get_robot_state(fsm.robot)
            fsm.run_once((manipulator_state, robot_state))

        step(pb, int(SIM_FREQUENCY/CONTROL_FREQUENCY))

        if all(fsm.current_state == "NONE" for fsm in fsms.values()):
            break


def step(pb, t):
    for _ in range(t):
        pb.stepSimulation()
        time.sleep(1. / SIM_FREQUENCY)


def get_cell_coordinates(x, y):
    return x + 0.5, y + 0.5


def get_agent(robot_id, pose, v):
    agent = Agent(robot_id)
    agent.position_ = get_vec(pose)
    agent.pref_velocity_ = get_vec(v)
    agent.velocity_ = get_vec(v)
    agent.id_ = robot_id
    agent.max_neighbors_ = 10
    agent.max_speed_ = utils.control.MAX_LINEAR_V
    agent.neighbor_dist_ = 5.0
    agent.radius_ = utils.control.RADIUS * 2
    agent.time_horizon_ = 4.0
    agent.time_horizon_obst_ = 10.0
    agent.time_step_ = 1 / CONTROL_FREQUENCY

    return agent


def update_pref_v(rvo, agentNo, v):
    rvo.set_agent_pref_velocity(agentNo, Vector2(v[0], v[1]))


def get_vec(coords):
    return Vector2(coords[0], coords[1])
