import numpy as np
import pybullet as pb

DEG_TO_RAD = np.pi/180

class SensorModels:
    def __init__(self, pb_client):
        self.pb_client = pb_client
        
    def laser_scans(self,
                    robot_id,
                    laser_max_range=10.0,
                    laser_min_range=.3,
                    laser_scanner_height=.1):
        pos, orien = self.pb_client.getBasePositionAndOrientation(bodyUniqueId=robot_id)

        ray_from_batch = [(pos[0]+laser_min_range*np.cos(DEG_TO_RAD*i),
                           pos[1]+laser_min_range*np.sin(DEG_TO_RAD*i),
                           laser_scanner_height) for i in range(360)]
        ray_to_batch = [(pos[0]+laser_max_range*np.cos(DEG_TO_RAD*i),
                         pos[1]+laser_max_range*np.sin(DEG_TO_RAD*i),
                         laser_scanner_height) for i in range(360)]

        result = self.pb_client.rayTestBatch(rayFromPositions=ray_from_batch,
                                             rayToPositions=ray_to_batch)

        hit_locations = []
        for hit_data in result:
            if hit_data[0] >= 0: hit_locations.append(hit_data[3])

        return hit_locations
