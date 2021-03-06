import numpy as np
import pybullet as p

from utils.robot_control import RobotControl
from utils.sensor_models import SensorModels

DEG_TO_RAD = np.pi/180
MAX_VOLUME = 100
OBJECT_STATUS = ["GROUND", "ASSIGNED", "REMOVED"]


def normalize_vector(v):
    return v/np.linalg.norm(v)


def in_tolerance(v1, v2):
    return np.linalg.norm(v1, v2) < 0.01


class ManipulatorStateMachine:

    def __init__(self, pb, robot_id):
        self.pb = pb
        self.handlers = {"LOWER": self.lower,
                         "GRAB": self.grab,
                         "PLACE": self.place,
                         "RELEASE": self.release,
                         "ORIGIN": self.origin}

        self.handler = self.origin
        self.current_state = "NONE"
        self.current_volume = 0  # TODO: Why did Shinkyu have this as randomly generated?

        self.robot = robot_id
        self.object = None
        self.constraint = None

        self.control = RobotControl(pb)
        self.target_state = [0, 0, 0, 0]

    def release(self):
        self.pb.removeConstraint(self.constraint)
        self.object = None
        self.constraint = None

        return "ORIGIN"

    def grab(self):
        self.constraint = self.pb.createConstraint(parentBodyUniqueId=self.robot,
                                                   parentLinkIndex=10,
                                                   childBodyUniqueId=self.object,
                                                   childLinkIndex=-1,
                                                   jointType=p.JOINT_FIXED,
                                                   jointAxis=[1, 0, 0],
                                                   parentFramePosition=[0.075, 0, 0],
                                                   childFramePosition=[0, 0, 0])

        return "PLACE"

    def lower(self, manipulator_state):
        target_state = np.array([-0.0016962233862355199, 1.2404879177129509, -0.901944524850455, 1.1624811955078364])

        if self.current_state == "LOWER":
            if in_tolerance(target_state, manipulator_state):
                return "GRAB"
        else:
            self.control.manipulator_control(self.robot, target_state)
            self.current_state = "LOWER"

        return "NONE"

    def place(self, manipulator_state):  # TODO: Why is the structure of place() different than lower()?
        if self.current_state == "PLACE":
            if in_tolerance(self.target_state, manipulator_state):
                return "RELEASE"
        else:
            self.target_state = np.array([np.pi - 0.2 * np.random.random(), 0, 0.4, 1.1])
            self.control.manipulator_control(self.robot, self.target_state)
            self.current_state = "PLACE"

        return "NONE"

    def origin(self, manipulator_state):
        target_state = np.array([5.08753421763152e-07, -1.3962637001751304, 1.0471974880549502, 0.523599283823225])

        if self.current_state == "ORIGIN":
            if in_tolerance(self.target_state, manipulator_state):
                return "DONE"
        else:
            self.control.manipulator_control(self.robot, target_state)
            self.current_state = "ORIGIN"

        return "NONE"

    def empty_basket(self):
        self.current_volume = 0

    def reinitialize(self):
        self.handler = self.lower
        self.current_state = "NONE"

    def run_once(self, manipulator_state):
        new_state = self.handler(manipulator_state)

        if new_state is "DONE":
            self.current_volume += 1

            if self.current_volume >= MAX_VOLUME:  # TODO: Why is this >=?
                return "RETRIEVE"

            return "DONE"

        elif new_state is not "NONE":
            self.handler = self.handlers[new_state]

        return "INPROCESS"


class RobotStateMachine:
    max_dest_timeout = 200
    max_esc_timeout = 80
    max_servo_timeout = 200
    collection_sites = [((0, 0), 5)]
    threshold_distance = 1.0
    gain_mag = 5.0
    gain_deg = 5.0

    def __init__(self, pb, robot_id, max_linear_v=0.2, max_rotational_v=5.0):
        self.pb = pb
        self.handlers = {"PICKUP": self.pickup,
                         "MOVE": self.move,
                         "SERVO": self.servo,
                         "RETRIEVE": self.retrieve}
        self.handler = self.move

        self.robot = robot_id
        self.destination = None
        self.world_state = None
        self.target_obj = None

        self.dest_dist = 0
        self.dest_timeout = 0
        self.esc_timeout = self.max_esc_timeout
        self.servo_timeout = 0

        self.arm_fsm = ManipulatorStateMachine(pb, robot_id)

        self.max_linear_v = max_linear_v
        self.max_rotational_v = max_rotational_v

        self.control = RobotControl(pb, max_linear_velocity=max_linear_v, max_rotational_velocity=max_rotational_v)

    def pickup(self, state):
        manipulator_state = state[0]
        result = self.arm_fsm.run_once(manipulator_state)

        if result is "DONE":
            try:
                self.world_state["objects"][self.target_obj][1] = "REMOVED"
            except KeyError:
                pass

            self.target_obj = None

            return "MOVE"

        elif result is "RETRIEVE":
            try:
                self.world_state["objects"][self.target_obj][1] = "REMOVED"
            except KeyError:
                pass

            self.target_obj = None

            return "RETRIEVE"

        return "NONE"

    def move(self, state):
        pose, vel = state[1]

        if self.esc_timeout < self.max_esc_timeout:
            self.esc_timeout += 1

        # if an object is detected, transition to visual servoing TODO: wtf does this mean
        if self.world_state is not None and self.esc_timeout >= self.max_esc_timeout:

            # find the object? this feels so inefficient TODO: understand this math
            for obj in self.world_state["objects"].keys():
                if self.world_state["objects"][obj][1] != "ON_GROUND": continue

                obj_pos = self.world_state["objects"][obj][0]
                obj_vector = np.array([obj_pos[0] - pose[0]])