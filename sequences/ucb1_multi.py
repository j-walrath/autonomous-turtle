import logging
from typing import Dict, List
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
K = 10       # Number of turtlebots (must be even)


def in_cell(target, state):
    return target[0] <= state[0] <= target[0]+1 and target[1] <= state[1] <= target[1]+1


# MULTI AGENT UCB TEST
def ucb1_multi(pb, objects: List[int], object_states: Dict[int, str], robots: List[int],
               robot_fsms: Dict[int, fsm.RobotStateMachine]):
    logging.info('Running Multi Agent UCB1...')

    # SIMULATION VARIABLES
    mu = [3, 2]                      # Center of object pile
    sig = [[0.4, 0], [0, 0.4]]       # Spread of objects
    var = 0.1                        # Measurement error

    field = np.zeros((M, M))         # Locations of objects
    visits = np.zeros((K, M, M))     # Visit history of the grid for each robot
    reward = np.zeros((K, M, M))     # Sum of rewards by location for each robot
    exp_mean = np.zeros((K, M, M))   # Expected mean reward by location for each robot

    # T = N                            # Number of time steps
    # delta = 1                        # Reward decreases by delta each visit
    xi = 2                           # Constant Xi > 1
    gamma = 1                        # Max message length

    msgTx = {}                       # Dict {agent: list of messages to send - tuple [agent, time, arm, reward]}
    msgRx = {}                       # Dict {agent: list of messages to received - tuple [agent, time, arm, reward]}

    regret = []

    controller = RobotControl(pb, robot_fsms)

    # INITIALIZE COMMUNICATION NETWORK
    logging.debug("Initializing communication network...")
    for agent in range(K):
        msgTx[agent] = []
        msgRx[agent] = []
        regret.append([0])

    # If graph[i][j] = 1 then there is an edge between agents i and j (graph must be symmetric)
    # graph = np.ones((K, K))
    degree = 9
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

    logging.info("Simulation Graph (DEGREE = {}, GAMMA = {}): \n{}".format(degree, gamma, graph))

    # LOAD OBJECTS AND ROBOTS
    logging.debug("Loading {} agents and {} objects...".format(K, N))
    coords = [(1, 7), (-2, 7), (-2, 4), (-2, 1), (-2, -2), (1, -2), (4, -2), (7, -2), (7, 1), (7, 4)]
    for robot in lib.load_robots(pb, coords[0:K]):
        robots.append(robot)
        robot_fsms[robot] = fsm.RobotStateMachine(pb, object_states, robot_fsms, robot)

    object_locations = np.random.multivariate_normal(mu, sig, N)

    for i, j in object_locations:
        x = math.floor(i) if i < 5 else 4
        y = math.floor(j) if j < 5 else 4
        field[x][y] += 1

    for obj in lib.load_objects(pb, object_locations):
        objects.append(obj)
        object_states[obj] = "ON_GROUND"

    lib.step(pb, 100)

    logging.debug("Executing Simulation...")

    # bot = robot_fsms[robots[0]]
    # fsms = bot.control.robot_fsms
    # logging.debug("{} is aware of fsms for {}".format(lib.NAMES[robots[0]-1],
    #                                                   [lib.NAMES[bot_id - 1] for bot_id in fsms]))
    # INITIALIZE AGENT
    logging.debug("Initializing agents...")
    T = 0
    for agent in range(K):
        robot = robots[agent]
        # Explore cells
        for x in range(M):
            for y in (range(M-1, -1, -1) if x & 1 else range(M)):
                robot_fsms[robot].set_destination(lib.get_cell_coordinates(x, y))
                lib.cycle_robot(pb, robot_fsms[robot])
                lib.step(pb, 10)

                logging.debug("{} measured cell ({}, {})".format(lib.NAMES[agent], x, y))
                visits[agent][x][y] += 1
                reward[agent][x][y] += controller.measure(robot, object_states, noise="GAUSSIAN", sigma=var)

        # Return to start
        robot_fsms[robot].set_destination(coords[agent])
        lib.cycle_robot(pb, robot_fsms[robot])
        # Calculate expected mean
        exp_mean[agent] = np.divide(reward[agent], visits[agent])

    # RUN UCB1
    T += 1
    while np.max(field) > 0:
        count = 0
        for obj in object_states.values():
            if obj in ("ON_GROUND", "ASSIGNED"):
                count += 1
        logging.debug("TIMESTEP T = {} ({} objects remain)".format(T, count))

        idxs = []

        for agent in range(K):
            robot = robots[agent]

            # Select arm with highest expected Q
            Q = exp_mean[agent] + var * np.divide(math.sqrt(2*(xi + 1)*math.log(T)), np.sqrt(visits[agent]))
            target = np.unravel_index(np.argmax(Q), Q.shape)

            # Make measurement
            robot_fsms[robot].set_destination(lib.get_cell_coordinates(target[0], target[1]))
            lib.cycle_robot(pb, robot_fsms[robot])
            lib.step(pb, 10)

            measurement = controller.measure(robot, object_states, noise="GAUSSIAN", sigma=var)
            idxs.append(target)

            logging.debug("{} measured {} at cell {}".format(lib.NAMES[agent], measurement, target))

            # Calculate regret
            max_reward = field[np.unravel_index(np.argmax(field), field.shape)] # based off optimal target
            actual_reward = field[target]
            regret[agent].append(max_reward - actual_reward)

            # Message neighbors
            msg = (T, agent, target, measurement)
            tx = msgTx[agent]
            if len(tx) == gamma:
                tx.pop(0)
            tx.append(msg)
            msgTx[agent] = tx

            # Return
            robot_fsms[robot].set_destination(coords[agent])
            lib.cycle_robot(pb, robot_fsms[robot])
            logging.debug("{} returned to their start position {}".format(lib.NAMES[agent], coords[agent]))

        for agent in range(K):
            robot = robots[agent]

            # Go to target
            Q = exp_mean[agent] + var * np.divide(math.sqrt(2 * (xi + 1) * math.log(T)), np.sqrt(visits[agent]))
            target = np.unravel_index(np.argmax(Q), Q.shape)
            robot_fsms[robot].set_destination(lib.get_cell_coordinates(target[0], target[1]))
            lib.cycle_robot(pb, robot_fsms[robot])

            # Pickup object
            flag = False
            for obj in objects:
                if object_states[obj] == "ON_GROUND" and in_cell(target, controller.get_object_state(obj)):
                    robot_fsms[robot].set_target(obj)
                    flag = True
                    break
            lib.cycle_robot(pb, robot_fsms[robot])

            logging.debug("{} {} an object at {}".format(lib.NAMES[agent], "picked up" if flag else "did not find",
                                                         target))

            # Decrement field
            if field[target] > 0:
                field[target] -= 1

            # Return to start location
            robot_fsms[robot].set_destination(coords[agent])
            lib.cycle_robot(pb, robot_fsms[robot])

            logging.debug("{} returned to their start location at {}".format(lib.NAMES[agent], coords[agent]))

        # Exchange messages and adjust expected mean
        for agent in range(K):
            new_visits = np.zeros((M, M))
            # Check new messages, skipping repeats
            for neighbor in range(K):
                if graph[agent][neighbor] == 1:
                    received = msgRx[agent]
                    sent = msgTx[neighbor]
                    for i in range(len(sent)):
                        msg = sent[i]
                        if received.count(msg) == 0:
                            received.append(msg)
                            cell = msg[2]
                            visits[agent][cell] += 1
                            new_visits[cell] += 1
                            reward[agent][cell] += msg[3]
                    msgRx[agent] = received

            # Calculate expected mean
            reward[agent] -= np.multiply(new_visits, visits[agent])
            exp_mean[agent] = np.divide(reward[agent], visits[agent])
        logging.debug("FIELD AT END OF TIMESTEP:\n{}".format(field))
        T += 1

    # FINAL OUTPUT
    logging.info("Simulation Complete!")
    cumulative_regret = np.cumsum(np.array(regret), axis=1)
    total_cumulative_regret = np.sum(cumulative_regret, axis=0)
    logging.info("Cumulative Regret:\n{}".format(cumulative_regret))
    plt.plot(np.arange(T), total_cumulative_regret)
    plt.show()

    with open("./outputs/Multi-Agent UCB d{}g{}.txt".format(degree, gamma), "a") as f:
        f.write("Multi-Agent UCB (Degree = {}, Message Passing {})\n\n".format(degree, "OFF" if gamma == 1 else "ON"))
        f.write("Cumulative Regret by Agent:\n")
        np.savetxt(f, cumulative_regret, fmt="%.2f", delimiter=", ")
        f.write("\n\nSystem Cumulative Regret:")
        np.savetxt(f, total_cumulative_regret, fmt="%.2f", delimiter=", ")
        f.write("\n\nTimesteps: {}".format(T))

    logging.debug("Returning to Main...")

    return
