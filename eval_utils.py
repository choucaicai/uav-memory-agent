
import numpy as np
from PIL import Image
import os, json
from dataclasses import dataclass
from collections import defaultdict 
import re
from pathlib import Path
import cv2  
import  warnings
from accelerate.utils import set_seed
from scipy.spatial.transform import Rotation as R
import numpy as np  
from typing import Dict,Optional,List
import io
from collections import defaultdict
import time
import math
import subprocess, threading
import airsim
from common import *

from runtime_config import load_runtime_config

RUNTIME_CONFIG = load_runtime_config()
AIRSIM_CONFIG_DIR = Path(RUNTIME_CONFIG['airsim_config_dir'])
import tqdm
import subprocess
import threading
import time
import os
import signal

ENV_DICT = RUNTIME_CONFIG['environments']
ENV_NAME_LIST = [
    "BrushifyUrban","CabinLake","CityPark","DownTown1","NewNeighborhood",
    "slum1","UrbanJapan","vinice_new1","WesternTown1","WinterTown1","Barnyard_test1","BrushifyRoad_test1",
    "BrushifyUrban","CleanCityStreet","DownTown_test1","NYC1950","Slum_test1","Vinice_test1","WesternTown_test1","WinterTown_test1"
]
class AirSimRunner:
    def __init__(self,env_name):
        self.processes = {}
        exe_path = ENV_DICT[env_name]
        self.base_command = [
            "bash", 
            # "ENVS/TRAIN_ENVS/BrushifyUrban/BrushifyUrban.sh",
            exe_path,
            "-RenderOffscreen",
            "-NoSound", 
            "-NoVSync"
        ]
        
    def run_single_env(self, gpu_id, settings_file, thread_id):
        """运行单个AirSim环境"""
        command = self.base_command + [
            f"-GraphicsAdapter={gpu_id}",
            f"--settings={settings_file}"
        ]
        print(f"Thread {thread_id}: Starting with GPU {gpu_id}, settings: {settings_file}")
        
        try:
            process = subprocess.Popen(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            self.processes[thread_id] = process
            
            # 等待一段时间检查启动状态
            time.sleep(5)
            
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"Thread {thread_id}: Process ended unexpectedly")
                print(f"stdout: {stdout}")
                print(f"stderr: {stderr}")
            else:
                print(f"Thread {thread_id}: AirSim started successfully")
                # 保持进程运行
                process.wait()
                
        except Exception as e:
            print(f"Thread {thread_id}: Error starting AirSim: {e}")
    
    def run_multiple_envs(self, config_list):
        """
        并行运行多个AirSim环境
        Args:
            config_list: 配置列表，每个元素为 (gpu_id, settings_file_path)
        """
        threads = []
        
        for i, (gpu_id, settings_file) in enumerate(config_list):
            thread = threading.Thread(
                target=self.run_single_env,
                args=(gpu_id, settings_file, i),
                daemon=True
            )
            threads.append(thread)
            thread.start()
            # 错开启动时间，避免资源冲突
            time.sleep(2)
        # 等待所有线程
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("Received interrupt signal, shutting down...")
            self.cleanup()
    
    def cleanup(self):
        """清理所有进程"""
        for thread_id, process in self.processes.items():
            try:
                if process.poll() is None:
                    print(f"Terminating process for thread {thread_id}")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass

class AirsimTrajRecorder:
    def __init__(self, env_name,airsim_port=30001,device_id=0,send_freq=10.0,enable_record_video=False):
        self.env_name = env_name
        self.airsim_port = airsim_port
        self.device_id = device_id

        self.air_runner = AirSimRunner(env_name=env_name)
        def run_air_sim(total_gpus = 4):
            air_configs = [
                (self.device_id  % total_gpus, self.change_and_save_settings(self.airsim_port))
            ]
            self.air_runner.run_multiple_envs(air_configs)
        
        self._sim_thread = threading.Thread(target=run_air_sim)
        self._sim_thread.start()
        
        time.sleep(10)

        # 初始化 AirSim 连接
        self._init_airsim_connection()
        if enable_record_video:
            while True:
                try:
                    self._client2 = airsim.MultirotorClient(port=self.airsim_port)
                    self._client2.confirmConnection()
                    print(f'AirSim client 2 connected successfully on port {self.airsim_port}!')
                    break
                except Exception as e:
                    print(f"连接 AirSim 失败: {e}")
                    time.sleep(2)
    def change_and_save_settings(self,air_port):
        base_settings = json.load(open(AIRSIM_CONFIG_DIR / 'base_settings_512.json'))
        base_settings["ApiServerPort"] = air_port
        new_path = str(AIRSIM_CONFIG_DIR / "eval_settings" / f"settings_512_{air_port}.json")
        os.makedirs(os.path.dirname(new_path),exist_ok=True)
        with open(new_path,'w') as f:
            json.dump(base_settings,f)
        return new_path
    def cleanup(self):
        self.air_runner.cleanup()
    def __del__(self):
        self.air_runner.cleanup()
        
    def _init_airsim_connection(self):
        """初始化 AirSim 连接"""
        while True:
            try:
                self._client = airsim.MultirotorClient(port=self.airsim_port)
                self._client.confirmConnection()
                self._client.enableApiControl(True)
                self._client.armDisarm(True)
                print(f'AirSim connected successfully on port {self.airsim_port}!')
                break
            except Exception as e:
                print(f"连接 AirSim 失败: {e}")
                time.sleep(2)
    # def set_camera_pose(self, x, y, z, pitch, yaw, roll):
    #     target_pose = airsim.Pose(airsim.Vector3r(x, -y, -z),
    #                               airsim.to_quaternion(math.radians(pitch), 0, math.radians(-yaw)))
    #     self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
    #     self._client.simSetVehiclePose(target_pose, True)
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

    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(pitch, -yaw, roll)
        target_pose = airsim.Pose(airsim.Vector3r(x, y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        time.sleep(0.1)

    def _camera_init(self):
        '''Camera initialization'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(15), 0, 0))
        self._client.simSetCameraPose("0", camera_pose)
        time.sleep(1)

    def _drone_init(self):
        '''Drone initialization'''
        self.set_drone_pos(0, 0, 0, 0, 0, 0)
        time.sleep(1)
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
    
    def get_camera_data(self, camera_type = 'color'):
        valid_types = {'color', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'color':
            image_type = airsim.ImageType.Scene
        elif camera_type == 'depth':
            image_type = airsim.ImageType.DepthPlanar
        else:
            image_type = airsim.ImageType.Segmentation

        responses = self._client.simGetImages([airsim.ImageRequest('front_custom', image_type, False, False)])
        response = responses[0]
        if response.pixels_as_float:
            img_data = np.array(response.image_data_float, dtype=np.float32)
            img_data = np.reshape(img_data, (response.height, response.width))
        else:
            img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_data = img_data.reshape(response.height, response.width, 3)

        return img_data
    
    
def calculate_distance_uavon(position, target_position):
    obs_pos = np.array(position)
    coords = np.array(target_position)
    # 如果 coords.shape = (N,3)，说明是多点
    if coords.ndim == 2 and coords.shape[1] == 3:
        dists = np.linalg.norm(coords - obs_pos[None, :], axis=1)
        return float(dists.min())
    else:
        return float(np.linalg.norm(obs_pos - coords))
    
    
def get_images(lst,if_his,step):
    if if_his is False:
        return lst[-1]
    else:
        if step == 1:
            if len(lst) >= 2:
                return [lst[-2], lst[-1]]
            elif len(lst) == 1:
                return [lst[0], lst[0]]
        elif step == 2:
            if len(lst) >= 3:
                return lst[-3:]
            elif len(lst) == 2:
                return [lst[0], lst[0], lst[1]]
            elif len(lst) == 1:
                return [lst[0],lst[0], lst[0]]

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

def load_eval_datasplit(eval_dataset,eval_samples_per_env=None):
    f = open(eval_dataset, 'r')
    all_eval_info = json.loads(f.read())
    f.close()
    ######## group  #########
    env_groups = defaultdict(list)
    for item in all_eval_info:
        # env_type = item["map_name"].split("/")[0]  # Get environment type
        env_type = item['map_name']
        env_groups[env_type].append(item)
    if eval_samples_per_env is not None:
        for env_name, eval_info in env_groups.items():
            env_groups[env_name] = eval_info[:eval_samples_per_env]
            
    print(f"all examples {len(all_eval_info)},all envs: {len(env_groups)},sample exampels: {sum([len(eval_info) for env_name, eval_info in env_groups.items()])}")
    
    return env_groups
def process_results(env_result_items_dict):
    default_seen = defaultdict(list)
    default_size = defaultdict(list)
    
    total_items = [item  for env,items in env_result_items_dict.items() for item in items]
    def cal_metric(total_rows):
        return {
            'num' : len(total_rows),
            'acc' : sum([cur['acc'] for cur in total_rows]) / len(total_rows) if len(total_rows) > 0 else 0,
            'osr' : sum([cur['osr'] for cur in total_rows]) / len(total_rows) if len(total_rows) > 0 else 0,
            'ne'  : sum([cur['ne'] for cur in total_rows]) / len(total_rows) if len(total_rows) > 0 else 0,
        }
    total_results = cal_metric(total_items)
    for row in total_items:
        size_str = row['size']
        size = re.search(r'^\s*([a-zA-Z]+)', size_str.strip()).group(1)
        is_seen = row["used-in-train"]
        default_seen[is_seen].append(row)
        default_size[size].append(row)

    seen_metric = {
        k:cal_metric(item_list) for k,item_list in default_seen.items()
    }
    size_metric = {
        k:cal_metric(item_list) for k,item_list in default_size.items()
    }
    return dict(
        total_results = total_results,
        seen_metric = seen_metric,
        size_metric = size_metric
    )
def print_results(results):
    """
    打印结果，分3行打印total_results, seen_metric, size_metric
    每行分4列打印cal_metric的指标（num, acc, osr, ne）
    """
    # 打印表头
    print(f"{'Category':<20} {'Num':<10} {'Acc':<10} {'OSR':<10} {'NE':<10}")
    print("-" * 60)
    
    # 打印 total_results
    total = results['total_results']
    print(f"{'Total':<20} {total['num']:<10} {total['acc']:<10.4f} {total['osr']:<10.4f} {total['ne']:<10.4f}")
    
    # 打印 seen_metric
    print("\n" + "=" * 60)
    print("Seen Metric:")
    print("-" * 60)
    for key, metrics in results['seen_metric'].items():
        label = f"  {key}"
        print(f"{label:<20} {metrics['num']:<10} {metrics['acc']:<10.4f} {metrics['osr']:<10.4f} {metrics['ne']:<10.4f}")
    
    # 打印 size_metric
    print("\n" + "=" * 60)
    print("Size Metric:")
    print("-" * 60)
    for key, metrics in results['size_metric'].items():
        label = f"  {key}"
        print(f"{label:<20} {metrics['num']:<10} {metrics['acc']:<10.4f} {metrics['osr']:<10.4f} {metrics['ne']:<10.4f}")
        
@dataclass
class EvalConfig:
    eval_dataset: Path = Path(RUNTIME_CONFIG['eval_dataset'])        # Path to Open-X dataset directory
    eval_samples_per_env: Optional[int] = None
    model_path :str= RUNTIME_CONFIG['model_path']
    eval_output_foler :Optional[str] = RUNTIME_CONFIG['output_root']
    eval_max_steps: int =100
    airsim_default_port: int = 30000
    
class EvalRunner():
    def __init__(self,config: Optional[EvalConfig] = None,seed: int=42, **kwargs):
        set_seed(42)
        if config is not None:
            self.cfg = config
        else:
            self.cfg = EvalConfig()
        # 用 kwargs 覆盖配置
        for key, value in kwargs.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
            else:
                warnings.warn(f"Init Eval Runner Unknown config parameter: '{key}' will be ignored", UserWarning)
        self.eval_groups = load_eval_datasplit(self.cfg.eval_dataset,self.cfg.eval_samples_per_env)
    def get_eval_groups(self):
        return self.eval_groups

    def __del__(self):
        kill_all_env_process()
        
    def run_eval(self,state,env_name,local_data,policy,processor):
        acc = 0
        stop = 0
        data_num = 0
        MAX_STEP = self.cfg.eval_max_steps
        DISTANCE_TO_SUCCESS = 20

        cur_env_result = []
        
        cur_airsim_port = int(self.cfg.airsim_default_port+state.process_index)
        airsim_env = AirsimTrajRecorder(env_name,airsim_port=cur_airsim_port,device_id=state.process_index)
        t_pbar=  tqdm.tqdm(enumerate(local_data),total=len(local_data),desc=f"Evaluating {env_name} On Thread {state.process_index}")
        for idx, item in t_pbar:
            acts = []  # Reset action list
            start_pose = item['start_pose']
            object_pose = item['pose']
            
            text = item['description']
            start_postion = start_pose["start_position"]
            start_quaternionr = start_pose["start_quaternionr"]
            
            pitch, roll, yaw = to_eularian_angles(start_quaternionr[0],start_quaternionr[1],start_quaternionr[2],start_quaternionr[3])
            start_yaw = yaw
            
            # print(f"Sample {idx}: {start_postion} -> {object_pose}, initial heading: {start_yaw}")
            gen_pose_list = []
            stop_error = 1
            image_error = False
            new_pose = [start_postion[0], start_postion[1], start_postion[2], start_yaw]
            
            # Set camera pose
            airsim_env._set_camera_pose(
                start_postion[0], 
                start_postion[1], 
                start_postion[2], 
                start_yaw,
                0, 
                0
            )
            
            image_list = []
            step = 0
            
            cur_osr = 0
            gen_pose_list.append((
                start_postion[0], 
                start_postion[1], 
                start_postion[2], 
                start_yaw, 
            ))
            while step < MAX_STEP:
                try:
                    raw_image_dict = airsim_env._capture_images(camera_names=['uav_on_0'],capture_depth=True)
                    raw_image = raw_image_dict['uav_on_0']['rgb']
                    cv2.imwrite("test_cur_img.jpg", raw_image)
                    image = raw_image
                    image_list.append(image)
                    model_action = get_action(policy, processor, image_list, text, acts, if_his=False, his_step=2,device=state.device)
                    acts.append(model_action)
                    new_pose = getPoseAfterMakeAction(new_pose, model_action)
                    # print(f'step:{step} action:',model_action,'position:',new_pose)
                    # print(f"Environment: {env_name}, Sample: {idx}, Step: {step}, Action: {model_action}, New position: {new_pose}")
                    airsim_env._set_camera_pose(
                        new_pose[0], 
                        new_pose[1], 
                        new_pose[2], 
                        new_pose[3], 
                        0, 
                        0
                    )
                    gen_pose_list.append(
                        (                        
                            new_pose[0], 
                            new_pose[1], 
                            new_pose[2], 
                            new_pose[3],
                        )
                    )
                    if model_action == 0:
                        stop_error = 0
                        break
                    step += 1
                    
                    if calculate_distance_uavon(new_pose[:3],object_pose) < 20.0:
                        cur_osr = 1
                        
                except Exception as e:
                    print(f"Error processing image: {e}")
                    image_error = True
                    break
                
            if image_error:
                continue
            
            model_end_position = new_pose
            dis = calculate_distance_uavon(model_end_position[:3],object_pose)
            if dis <= 20:
                acc += 1        
                
            stop += stop_error
            data_num += 1
            # print(f"Current accuracy: {acc/data_num:.4f}, Stop rate: {1-stop/data_num:.4f}, Evaluated: {data_num} samples")
            t_pbar.set_postfix_str(f"total acc:{acc/data_num:.2f},cur acc:{sum([cur['acc'] for cur in cur_env_result]) / len(cur_env_result) if len(cur_env_result) > 0 else 0}")
            cur_res = {}
            cur_res['size'] = item['size']
            cur_res['used-in-train'] = item['used-in-train']
            cur_res['acc'] = int(dis <= 20)
            cur_res['osr'] = cur_osr
            cur_res['ne'] = dis
            cur_res['gen_pose_list'] = gen_pose_list
            
            cur_env_result.append(cur_res)
            
        airsim_env.cleanup()
        del airsim_env
        import gc
        gc.collect()
        
        return cur_env_result
                


    
def kill_env_process(keyword):
    try:
        result = subprocess.run(
            ['pkill', '-9', '-f', keyword],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Successfully killed processes matching '{keyword}'")
            return True
        elif result.returncode == 1:
            print(f"No processes found matching '{keyword}'")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def kill_all_env_process():
    
    for env_name in [
            "BrushifyUrban","CabinLake","CityPark","DownTown1","NewNeighborhood",
            "slum1","UrbanJapan","vinice_new1","WesternTown1","WinterTown1","Barnyard_test1","BrushifyRoad_test1",
            "BrushifyUrban","CleanCityStreet","DownTown_test1","NYC1950","Slum_test1","Vinice_test1","WesternTown_test1","WinterTown_test1"
        ]:
        kill_env_process(env_name)
