import time

OBJ_MODEL = "./urdf_models/objects/object.urdf"
ROBOT_MODEL = "./urdf_models/tb_openmanipulator/trash_collect_robot_four_wheel.urdf"
PLANE_MODEL = "./urdf_models/plane_with_dumpsters.urdf"

SIM_FREQUENCY = 240


def load_plane(pb, position=None, lateralFriction=3.0, spinningFriction=0.03, rollingFriction=0.03, restitution=0.5,
               scaling=1.0):
    if position is None:
        position = [0, 0, 0]

    plane = pb.loadURDF(PLANE_MODEL, basePosition=position, globalScaling=scaling)
    pb.changeDynamics(0, -1, lateralFriction=lateralFriction, spinningFriction=spinningFriction,
                      rollingFriction=rollingFriction, restitution=restitution)

    return plane


def load_objects(pb, locations, COLLISION=False):
    objects = []
    for loc in locations:
        objects.append(pb.loadURDF(OBJ_MODEL, basePosition=[loc[0], loc[1], 0.3], globalScaling=1.0))

        if not COLLISION:
            # Do not collide with robots or other objects
            pb.setCollisionFilterGroupMask(objects[-1], -1, 0, 0)

            # Do collide with the ground plane
            pb.setCollisionFilterPair(objects[-1], 0, -1, -1, 1)

    return objects


def load_robots(pb, locations):
    robots = []
    orn = pb.getQuaternionFromEuler([0, 0, 0])
    for loc in locations:
        robots.append(pb.loadURDF(ROBOT_MODEL, basePosition=[loc[0], loc[1], 0.5], baseOrientation=orn))
        pb.changeDynamics(robots[-1], -1, maxJointVelocity=300, lateralFriction=1.0, rollingFriction=0.03,
                          restitution=0.7)
    return robots


def step(pb, t):
    for _ in range(t):
        pb.stepSimulation()
        time.sleep(1. / SIM_FREQUENCY)
