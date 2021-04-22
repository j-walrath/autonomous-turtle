import logging
from utils import simulator_library as lib
from utils.control import RobotControl
from utils import fsm

import numpy as np
import random
import math
import matplotlib.pyplot as plt

CONTROL_TIME = int(lib.SIM_FREQUENCY/lib.SIM_FREQUENCY)
N = 100     # Number of objects
M = 5       # Size of grid
K = 6       # Number of turtlebots (must be even)


def in_cell(target, state):
    return target[0] <= state[0] <= target[0]+1 and target[1] <= state[1] <= target[1]+1


# SINGLE AGENT UCB TEST
def ucb1_multi(pb, objects: list, object_states: dict, robots: list, robot_fsms: list):
    logging.info('Running Multi Agent UCB1...')

    # SIMULATION VARIABLES
    mu = [3, 2]                      # Center of object pile
    sig = [[0.4, 0], [0, 0.4]]       # Spread of objects
    var = 0.1                        # Measurement error

    field = np.zeros((M, M))         # Locations of objects
    visits = np.zeros((K, M, M))     # Visit history of the grid for each robot
    reward = np.zeros((K, M, M))     # Sum of rewards by location for each robot
    exp_mean = np.zeros((K, M, M))   # Expected mean reward by location for each robot

    T = N                            # Number of time steps
    delta = 1                        # Reward decreases by delta each visit
    xi = 2                           # Constant Xi > 1
    gamma = 3                        # Max message length

    msgTx = {}                       # Dict {agent: list of messages to send - tuple [agent, time, arm, reward]}
    msgRx = {}                       # Dict {agent: list of messages to received - tuple [agent, time, arm, reward]}

    regret = []

    controller = RobotControl(pb)

    # INITIALIZE COMMUNICATION NETWORK
    for i in range(K):
        msgTx[i] = []
        msgRx[i] = []
        regret.append([0])

    # If graph[i][j] = 1 then there is an edge between agents i and j (graph must be symmetric)
    # graph = np.ones((K, K))
    degree = 4
    graph = np.identity(K)
    if degree % 2 == 0:
        count = int(degree/2)
        for j in range(count):
            # Make vertex for every (i + 1) steps away
            for i in range(K):
                neighbor = int(i + j + 1)
                if neighbor > (K - 1):
                    neighbor = neighbor - K
                graph[i][neighbor] = 1
                graph[neighbor][i] = 1
    else:
        count = int((degree - 1)/2 + 1)
        for j in range(count):
            # Make vertex for every n/2 - i steps away
            for i in range(K):
                neighbor = int(i + K/2 - j)
                if neighbor > (K - 1):
                    neighbor = neighbor - K
                graph[i][neighbor] = 1
                graph[neighbor][i] = 1

    logging.info("Simulation Graph: {}".format(graph))

    # LOAD OBJECTS AND ROBOTS
    coords = []
    for i in range(K):
        coords.append([math.ceil(i / 2.) - 1, -1] if i & 1 else [-1, math.ceil(i / 2.) - 1])

    for robot in lib.load_robots(pb, coords):
        robots.append(robot)
        robot_fsms[robot] = fsm.RobotStateMachine(pb, object_states, robot)

    object_locations = np.random.multivariate_normal(mu, sig, N)

    for obj in lib.load_objects(pb, object_locations):
        objects.append(obj)
        object_states[obj] = "ON_GROUND"

    logging.debug("Executing Simulation...")

    # INITIALIZE AGENT
    for agent in range(K):
        robot = robots[agent]
        # Explore cells
        for x in range(M):
            for y in (range(M-1, -1, -1) if x & 1 else range(M)):
                robot_fsms[robot].set_destination(lib.get_cell_coordinates(x, y))
                lib.cycle_robot(robot_fsms[robot])
                lib.step(pb, 100)

                logging.debug("Robot {} measured cell ({}, {})".format(robot, x, y))
                visits[agent][x][y] += 1
                reward[agent][x][y] += controller.measure(robot, object_states, noise="GAUSSIAN", sigma=var)

        # Return to start
        robot_fsms[robot].set_destination(coords[agent])
        lib.cycle_robot(robot_fsms[robot])
        # Calculate expected mean
        exp_mean[agent] = np.divide(reward[agent], visits[agent])

    # RUN UCB1
    T += 1
    while np.max(field) > 0:
        logging.debug("TIMESTEP T = {}".format(T))
        idxs = []

        for agent in range(K):
            robot = robots[agent]

            # Select arm with highest expected Q
            Q = exp_mean[agent] + var * np.divide(math.sqrt(2*(xi + 1)*math.log(T)), np.sqrt(visits[agent]))
            target = np.unravel_index(np.argmax(Q), Q.shape)

            # Make measurement
            robot_fsms[robot].set_destination(lib.get_cell_coordinates(target[0], target[1]))
            lib.cycle_robot(robot_fsms[robot])
            lib.step(pb, 20)

            measurement = controller.measure(robot, objects, noise="GAUSSIAN", sigma=var)
            idxs.append(target)

            # Calculate regret
            max_reward = field[np.unravel_index(np.argmax(field), field.shape)] # based off optimal target
            reward = field[target]
            regret[agent].append(max_reward - reward)

            # Message neighbors
            msg = (T, agent, target, measurement)
            tx = msgTx[agent]
            if len(tx) == gamma:
                tx.pop(0)
            tx.append(msg)
            msgTx[agent] = tx

            # Pickup object (TODO: Check with Gargi that this is in the right place.)
            for obj in objects:
                if object_states[obj] == "ON_GROUND" and in_cell(target, controller.get_object_state(obj)):
                    robot_fsms[robot].set_target(obj)
                    break
            lib.cycle_robot(robot_fsms[robot])

            # Return to start location
            robot_fsms[robot].set_destination(coords[agent])
            lib.cycle_robot(robot_fsms[robot])

        # Exchange messages and adjust expected mean
        for agent in range(K):
            new_visits = np.zeros((M, M))
            # Check new messages, skipping repeats
            for i in range(K):
                if graph[agent][i] == 1:
                    received = msgRx[agent]
                    sent = msgTx[i]
                    for msg in sent:
                        if received.count(msg) == 0:
                            received.append(msg)
                            idx = msg[2]
                            visits[agent][idx] += 1
                            new_visits[idx] += 1
                            reward[agent][idx] += msg[3]
                    msgRx[agent] = received

            # Calculate expected mean
            reward[agent] -= np.multiply(new_visits, visits[agent])
            exp_mean[agent] = np.divide(reward[agent], visits[agent])

        T += 1

    # FINAL OUTPUT
    logging.info("Simulation Complete!")
    cumulative_regret = np.cumsum(np.array(regret), axis=1)
    logging.info("Cumulative Regret: {}".format(cumulative_regret))
    plt.plot(np.arange(T), cumulative_regret)

    logging.debug("Returning to Main...")

    return
