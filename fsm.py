import numpy as np
import pybullet as p

from control import RobotControl

DEG_TO_RAD = np.pi/180
MAX_VOLUME = 1
DISTANCE_THRESHOLD = 0.3
OBJECT_STATUS = ["ON_GROUND", "ASSIGNED", "RECOVERED", "RETRIEVED"]


def normalize_vector(v):
    return v/np.linalg.norm(v)


def in_tolerance(v1, v2):
    return np.linalg.norm(v1 - v2) < 0.01


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
        self.current_volume = 0

        self.robot = robot_id
        self.object = None
        self.constraint = None

        self.control = RobotControl(pb)
        self.target_state = np.array([0, 0, 0, 0])

    def release(self, manipulator_state):
        self.pb.removeConstraint(self.constraint)
        self.object = None
        self.constraint = None

        self.current_state = "ORIGIN"
        return "ORIGIN"

    def grab(self, manipulator_state):
        self.constraint = self.pb.createConstraint(parentBodyUniqueId=self.robot,
                                                   parentLinkIndex=10,
                                                   childBodyUniqueId=self.object,
                                                   childLinkIndex=-1,
                                                   jointType=p.JOINT_FIXED,
                                                   jointAxis=[1, 0, 0],
                                                   parentFramePosition=[0.075, 0, 0],
                                                   childFramePosition=[0, 0, 0])
        self.current_state = "PLACE"
        return "PLACE"

    def lower(self, manipulator_state):
        target_state = np.array([-0.0016962233862355199, 1.2404879177129509, -0.901944524850455, 1.1624811955078364])

        if self.current_state == "LOWER" and in_tolerance(target_state, manipulator_state[0][:4]):
            self.current_state = "GRAB"
            return "GRAB"
        else:
            self.control.manipulator_control(self.robot, target_state)
            self.current_state = "LOWER"

        return "NONE"

    def place(self, manipulator_state):  # TODO: Why is the structure of place() different than lower()?
        if self.current_state == "PLACE" and in_tolerance(self.target_state, manipulator_state[0][:4]):
            self.current_state = "RELEASE"
            return "RELEASE"
        else:
            self.target_state = np.array([np.pi - 0.2 * np.random.random(), 0, 0.4, 1.1])
            self.control.manipulator_control(self.robot, self.target_state)
            self.current_state = "PLACE"

        return "NONE"

    def origin(self, manipulator_state):
        target_state = np.array([5.08753421763152e-07, -1.3962637001751304, 1.0471974880549502, 0.523599283823225])

        if self.current_state == "ORIGIN" and in_tolerance(self.target_state, manipulator_state[0][:4]):
            self.current_state = "DONE"
            return "DONE"
        else:
            self.control.manipulator_control(self.robot, target_state)
            self.current_state = "ORIGIN"

        return "NONE"

    def reinitialize(self):
        self.handler = self.lower
        self.current_state = "NONE"

    def run_once(self, manipulator_state):
        new_state = self.handler(manipulator_state)

        if new_state is "DONE":
            self.current_volume += 1

            if self.current_volume >= MAX_VOLUME:
                self.current_state = "RETRIEVE"
                return "RETRIEVE"

            self.current_state = "DONE"
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

    def __init__(self, pb, obj_states, robot_id, max_linear_v=0.2, max_rotational_v=5.0):
        self.pb = pb
        self.handlers = {"PICKUP": self.pickup,
                         "MOVE": self.move,
                         "VISUALSERVO": self.visual_servo,
                         "RETRIEVE": self.retrieve}
        self.handler = self.move

        self.robot = robot_id
        self.current_state = "NONE"
        self.destination = None
        self.target_obj = None
        self.obj_states = obj_states
        self.basket = set()

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

        if result is not "INPROCESS":
            try:
                self.obj_states[self.target_obj] = "RECOVERED"
            except KeyError:
                pass

            self.basket.add(self.target_obj)
            self.target_obj = None

            if result is "DONE":
                self.current_state = "MOVE"
                return "MOVE"

            elif result is "RETRIEVE":
                self.current_state = "RETRIEVE"
                return "RETRIEVE"

        self.current_state = "NONE"
        return "NONE"

    def move(self, state):
        pose, vel = state[1]

        if self.esc_timeout < self.max_esc_timeout:
            self.esc_timeout += 1

        # if an object is detected, transition to visual servoing
        if self.obj_states is not None and self.esc_timeout >= self.max_esc_timeout:

            # find the closest object (this feels so inefficient) TODO: make this faster
            for obj in self.obj_states.keys():
                if self.obj_states[obj] != "ON_GROUND": continue

                obj_pos, orn = self.pb.getBasePositionAndOrientation(bodyUniqueId=obj)
                obj_vector = np.array([obj_pos[0] - pose[0], obj_pos[1] - pose[1]])

                if np.linalg.norm(obj_vector) <= 0.5:
                    self.control.velocity_control(self.robot, 0, 0)
                    self.target_obj = obj
                    self.obj_states[obj] = "ASSIGNED"
                    self.servo_timeout = 0
                    self.current_state = "VISUALSERVO"
                    return "VISUALSERVO"

        if self.destination is not None:
            dist = np.linalg.norm((self.destination[0] - pose[0], self.destination[1] - pose[1]))

            # increase the timeout counter when there is no improvement in reaching the destination
            if self.dest_dist <= dist:
                self.dest_timeout += 1

                if self.dest_timeout >= self.max_dest_timeout:
                    self.control.velocity_control(self.robot, 0, 0)
                    self.set_destination(None)

                    self.current_state = "NONE"
                    return "NONE"

            self.dest_dist = dist

            if self.dest_dist > DISTANCE_THRESHOLD:
                vl, vr = self.control.pose_control(self.robot, self.destination)

            else:
                self.control.velocity_control(self.robot, 0, 0)
                self.set_destination(None)

        self.current_state = "NONE"
        return "NONE"

    def visual_servo(self, state):
        pose, vel = state[1]

        try:
            if self.obj_states[self.target_obj] in ("REMOVED", "RETRIEVED"):
                self.target_obj = None
                self.current_state = "MOVE"
                return "MOVE"

            obj_pos = self.control.get_object_state(self.target_obj)

        except KeyError:
            self.target_obj = None
            self.current_state = "MOVE"
            return "MOVE"

        dist, orn, vl, vr = self.control.visual_servoing(self.robot, obj_pos, pose)

        if dist <= 0.05 and orn < 10 * np.pi / 180:
            self.control.velocity_control(self.robot, 0, 0)
            self.arm_fsm.reinitialize()
            self.arm_fsm.object = self.target_obj
            self.current_state = "PICKUP"
            return "PICKUP"

        self.current_state = "NONE"
        return "NONE"

    def retrieve(self, state):
        pose, vel = state[1]

        return_site = [0, 0]
        return_dist = np.linalg.norm((return_site[0] - pose[0], return_site[1] - pose[1]))

        if return_dist > DISTANCE_THRESHOLD:
            self.control.pose_control(self.robot, return_site)
        else:
            self.control.velocity_control(self.robot, 0, 0)
            self.empty_basket()
            self.set_destination(None)
            self.current_state = "MOVE"
            return "MOVE"

        self.current_state = "NONE"
        return "NONE"

    def set_destination(self, destination):
        self.destination = destination
        self.dest_timeout = 0

    def empty_basket(self):
        while self.basket:
            obj = self.basket.pop()
            self.obj_states[obj] = "RETRIEVED"
            self.pb.removeBody(obj)
        self.arm_fsm.current_volume = 0

    def run_once(self, state):
        new_state = self.handler(state)
        if new_state is not "NONE":
            self.handler = self.handlers[new_state]

    def breakdown(self):
        self.control.velocity_control(self.robot, 0, 0)

        return

