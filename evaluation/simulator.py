"""Observation, pose, and action helpers used by the reference evaluator."""
import io
import math
from pathlib import Path

import airsim
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from runtime_config import load_runtime_config

RUNTIME_CONFIG = load_runtime_config()
AIRSIM_CONFIG_DIR = Path(RUNTIME_CONFIG['airsim_config_dir'])
ENV_DICT = RUNTIME_CONFIG['environments']


class AirsimTrajRecorder:
    def _set_camera_pose(self, x, y, z, yaw, pitch, roll):
        """设置相机位姿"""
        # 停止移动
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        # 转换欧拉角到四元数
        x_val, y_val, z_val, w_val  = R.from_euler('ZYX', [yaw, pitch, roll]).as_quat()
        # 设置位姿
        target_pose = airsim.Pose()
        target_pose.position.x_val = float(x)
        target_pose.position.y_val = float(y)
        target_pose.position.z_val = float(z)

        target_pose.orientation.w_val = w_val
        target_pose.orientation.x_val = x_val
        target_pose.orientation.y_val = y_val
        target_pose.orientation.z_val = z_val

        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)


    def _capture_images(self, camera_names=["0"],capture_depth=False,):
        """捕获并保存图像"""
        img_dict = {}
        for camera_name in camera_names:
            req_list = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene)]
            if capture_depth:
                req_list.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True,False))
            camera_responses = self._client.simGetImages(req_list)

            rgb_response = camera_responses[0]
            rgb_img_buffer = io.BytesIO(rgb_response.image_data_uint8)
            rgb_img = Image.open(rgb_img_buffer)
            rgb_img = np.array(rgb_img)

            camera_info = self._client.simGetCameraInfo(camera_name)
            metainfo = {
                "fov": camera_info.fov,
            }

            if capture_depth:
                depth_response = camera_responses[1]
                depth_img = np.array(depth_response.image_data_float, dtype=np.float32).reshape(
                            depth_response.height, depth_response.width)

                quat_wb =  depth_response.camera_orientation
                world_pos = depth_response.camera_position
                metainfo['quat_wb'] = [quat_wb.x_val, quat_wb.y_val, quat_wb.z_val, quat_wb.w_val]
                metainfo['camera_pos'] = [world_pos.x_val, world_pos.y_val, world_pos.z_val]

            else:
                depth_img = None

            img_dict[camera_name] = {
                'depth': depth_img if depth_img is not None else None,
                'rgb': rgb_img,
                'metainfo': metainfo,
            }
        return img_dict


def calculate_distance_uavon(position, target_position):
    obs_pos = np.array(position)
    coords = np.array(target_position)
    # 如果 coords.shape = (N,3)，说明是多点
    if coords.ndim == 2 and coords.shape[1] == 3:
        dists = np.linalg.norm(coords - obs_pos[None, :], axis=1)
        return float(dists.min())
    else:
        return float(np.linalg.norm(obs_pos - coords))


def convert_to_action_id(action):
    action_dict = {
        "0": np.array([1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # stop
        "1": np.array([0, 3, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward
        "2": np.array([0, 0, 15, 0, 0, 0, 0, 0]).astype(np.float32),  # turn left 30
        "3": np.array([0, 0, 0, 15, 0, 0, 0, 0]).astype(np.float32),  # turn right 30
        "4": np.array([0, 0, 0, 0, 2, 0, 0, 0]).astype(np.float32),  # go up
        "5": np.array([0, 0, 0, 0, 0, 2, 0, 0]).astype(np.float32),  # go down
        "6": np.array([0, 0, 0, 0, 0, 0, 5, 0]).astype(np.float32),  # move left
        "7": np.array([0, 0, 0, 0, 0, 0, 0, 5]).astype(np.float32),  # move right

        "8": np.array([0, 6, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 6
        "9": np.array([0, 9, 0, 0, 0, 0, 0, 0]).astype(np.float32),  # move forward 9
    }
    action_values = list(action_dict.values())
    result = 0
    matched = False
    for idx, value in enumerate(action_values):
        if np.array_equal(action, value):
            result = idx
            matched = True
            break
    # If no match is found, default to 0
    if not matched:
        result = 0
    return result


def getPoseAfterMakeAction(new_pose, action):
    x, y, z, yaw = new_pose

    # Define step size
    step_size = 3.0  # Translation step size (units can be adjusted as needed)

    # Update new_pose based on action value
    if action == 0:
        pass
    elif action == 1:
        x += step_size * math.cos(yaw)
        y += step_size * math.sin(yaw)
    elif action == 2:
        yaw += math.radians(30)
    elif action == 3:
        yaw -= math.radians(30)
    elif action == 4:
        z += step_size
    elif action == 5:
        z -= step_size
    elif action == 6:
        x -= step_size * math.sin(yaw)
        y += step_size * math.cos(yaw)
    elif action == 7:
        x += step_size * math.sin(yaw)
        y -= step_size * math.cos(yaw)
    elif action == 8:
        x += step_size * math.cos(yaw) *2
        y += step_size * math.sin(yaw) *2
    elif action == 9:
        x += step_size * math.cos(yaw) *3
        y += step_size * math.sin(yaw) *3

    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

    return [x, y, z, yaw]


def to_eularian_angles(x_val = 0.0, y_val = 0.0, z_val = 0.0, w_val = 1.0):
    z = z_val
    y = y_val
    x = x_val
    w = w_val
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)
    return (pitch, roll, yaw)
