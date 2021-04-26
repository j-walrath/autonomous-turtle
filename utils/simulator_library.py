import time
from argparse import ArgumentError
from typing import List, Dict

import numpy as np
import pybullet as p

import utils.control
from utils.fsm import RobotStateMachine
from utils.control import RobotControl
from utils.rvo import rvo_math
from utils.rvo.agent import Agent
from utils.rvo.kdtree import KdTree
from utils.rvo.vector import Vector2
from utils.rvo.obstacle import Obstacle

OBJ_MODEL = "./urdf_models/objects/object.urdf"
ROBOT_MODEL = "./urdf_models/tb_openmanipulator/trash_collect_robot_four_wheel.urdf"
PLANE_MODEL = "./urdf_models/plane_with_dumpsters.urdf"

SIM_FREQUENCY = 240
CONTROL_FREQUENCY = 40
NAMES = ["Fluffy", "Oogway", "Crush", "Franklin", "Genbu", "Yertle", "Leonardo", "Raphael", "Donatello", "Michelangelo"]

ROBOT_RADIUS = 0.2
OBSTACLE_RADIUS = 0.3


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


def normalize_angle(theta):
    if theta > np.pi:
        theta -= 2 * np.pi
    elif theta <= -np.pi:
        theta += 2 * np.pi

    return theta


def get_effective_center(pose, D=ROBOT_RADIUS):
    x, y, yaw = pose
    return x + D * np.cos(yaw), y + D * np.sin(yaw)


def M(theta, D=ROBOT_RADIUS):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -D*s],
                     [s, D*c]])


def convert_v(pose, v, w):
    return np.dot(M(pose[2]), (v, w))


def revert_v(pose, vx, vy):
    return np.linalg.solve(M(pose[2]), (vx, vy))


def get_agent(robot_id, pose, v):
    virtual_center = get_effective_center(pose)
    center_v = convert_v(pose, np.linalg.norm(v[0:2]), v[2])

    agent = Agent(robot_id)
    agent.position_ = get_vec(virtual_center)
    agent.pref_velocity_ = get_vec(center_v)
    agent.velocity_ = get_vec(center_v)
    agent.max_neighbors_ = 10
    agent.max_speed_ = utils.control.MAX_LINEAR_V
    agent.neighbor_dist_ = 2.0
    agent.radius_ = ROBOT_RADIUS * 2
    agent.time_horizon_ = 5.0
    agent.time_horizon_obst_ = 5.0
    agent.time_step_ = 1 / CONTROL_FREQUENCY

    return agent


def get_obstacle_list(obstacle_poses, r=OBSTACLE_RADIUS):

    obstacle_vertices = []
    for (x, y, yaw) in obstacle_poses:
        obstacle_vertices.append([[x + r * np.cos(yaw + np.pi / 4), y + r * np.sin(yaw + np.pi / 4)],
                                  [x + r * np.cos(yaw + 3 * np.pi / 4), y + r * np.sin(yaw + 3 * np.pi / 4)],
                                  [x + r * np.cos(yaw + 5 * np.pi / 4), y + r * np.sin(yaw + 5 * np.pi / 4)],
                                  [x + r * np.cos(yaw + 7 * np.pi / 4), y + r * np.sin(yaw + 7 * np.pi / 4)]])

    obstacles = []
    for vertices in obstacle_vertices:
        vec_vertices = [get_vec(vertex) for vertex in vertices]
        add_obstacle(obstacles, vec_vertices)
    return obstacles


def add_obstacle(obstacle_list, vertices):
    if len(vertices) < 2:
        raise ArgumentError('Must have at least 2 vertices.')

    obstacleNo = len(obstacle_list)

    for i in range(len(vertices)):
        obstacle = Obstacle()
        obstacle.point_ = vertices[i]

        if i != 0:
            obstacle.previous_ = obstacle_list[-1]
            obstacle.previous_.next_ = obstacle

        if i == len(vertices) - 1:
            obstacle.next_ = obstacle_list[obstacleNo]
            obstacle.next_.previous_ = obstacle

        obstacle.direction_ = rvo_math.normalize(
            vertices[0 if i == len(vertices) - 1 else i + 1] - vertices[i])

        if len(vertices) == 2:
            obstacle.convex_ = True
        else:
            obstacle.convex_ = rvo_math.left_of(
                vertices[len(vertices) - 1 if i == 0 else i - 1],
                vertices[i],
                vertices[0 if i == len(vertices) - 1 else i + 1]) >= 0.0

        obstacle.id_ = len(obstacle_list)
        obstacle_list.append(obstacle)
    return obstacleNo


def build_trees(agents, obstacles):
    kd_tree = KdTree(agents=agents, obstacles=obstacles)
    kd_tree.build_agent_tree()
    kd_tree.build_obstacle_tree()
    return kd_tree


def get_vec(coords):
    return Vector2(coords[0], coords[1])
