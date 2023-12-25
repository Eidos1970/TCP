import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
import yaml
import copy

import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent

from TCP.model import TCP
from TCP.config import GlobalConfig
from team_code.planner import RoutePlanner

# <========================================================================
import sys
top_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if not top_path in sys.path:
    sys.path.append(top_path)

with open('./config/tcp_config.yml') as f:
    content = f.read()
    dic_path = yaml.load(content, Loader=yaml.SafeLoader)
    print('dic_path')
    print(dic_path)

with open('./config/com_params.yml') as f:
    content = f.read()
    com_params = yaml.load(content, Loader=yaml.SafeLoader)
    print('SNR: %f dB'%com_params['jscc']['snr_db'])

# Add the top level directory in system path
top_path_tcp_jscc = dic_path['rootPath_TCP_JSCC']
if not top_path_tcp_jscc in sys.path:
    sys.path.append(top_path_tcp_jscc)

from tools.common_tools import info_show
from models.svae.svae_model import SoftIntroVAE
from models.channel.channel_network import ChannelCodec
from models.channel.channel_physical import Channels
from models.jpeg.jpeg_model import JPEG
from models.j2k.j2k_model import J2K
from models.tradicom.tradicom_model import TradiCom, power_norm
from models.bpg.bpg_model import BPG
from tools.dataset_tcp import NormalizeManager
from tools.common_tools import reparameterize
from models.ae_ch.ae_model import AE
# from pythae_ex.models import AutoModel_Ex
# typing
from typing import Dict, List, Tuple, Union, Callable, Iterable, Any

PATH_VAE_MODEL = os.environ.get('PATH_VAE_MODEL', None)
PATH_CH_MODEL =  os.environ.get('PATH_CH_MODEL', None)  
SAVE_PATH = os.environ.get('SAVE_PATH', None)
TCP_PERCEPTION = os.environ.get('TCP_PERCEPTION', None)
TCP_MEASUREMENT = os.environ.get('TCP_MEASUREMENT', None)
MODEL_TYPE = os.environ.get('MODEL_TYPE', None)
MODE_PRECISION = os.environ.get('MODE_PRECISION', None)

QUALITY = int(os.environ.get('QUALITY', None))

USE_WANDB = os.environ.get('USE_WANDB', None)
if USE_WANDB == 'True':
    import wandb
# ========================================================================>

def get_entry_point():
    return 'TCPAgent'


class TCPAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.alpha = 0.3
        self.status = 0
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = deque()

        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.config = GlobalConfig()
        self.net = TCP(self.config)


        ckpt = torch.load(path_to_conf_file)
        ckpt = ckpt["state_dict"]
        new_state_dict = OrderedDict()
        for key, value in ckpt.items():
            new_key = key.replace("model.","")
            new_state_dict[new_key] = value
        self.net.load_state_dict(new_state_dict, strict = False)
        self.net.cuda()
        self.net.eval()

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0

        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

        self.last_steers = deque()
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print (string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
        
        # <====================================================================
        # Build the codec
        if MODEL_TYPE is not None:
            if MODEL_TYPE == 'JPEG':
                self.codec = JPEG()
            elif MODEL_TYPE == 'J2K':
                self.codec = J2K()
            elif MODEL_TYPE == 'BPG':
                self.codec = BPG()
            elif MODEL_TYPE == 'JSCC' or MODEL_TYPE == 'AE':
                self.device = torch.device('cuda:0')
                if MODEL_TYPE == 'JSCC':
                    self.codec = SoftIntroVAE(cdim=3, zdim=com_params['jscc']['zdim'], 
                                                channels=[64, 128, 256, 512, 512, 512], 
                                                image_size=(256,900))
                else:
                    self.codec = AE(cdim=3, zdim=1024, 
                                channels=[64, 128, 256, 512, 512, 512], 
                                image_size=(256,900))
                self.codec.to(self.device)
                weights = torch.load(PATH_VAE_MODEL, map_location=self.device)
                self.codec.load_state_dict(weights['model'], strict=False)
                self.codec.eval()
                self.norm_manager = NormalizeManager()
            elif MODEL_TYPE == 'ORIGIN':
                pass
            else:
                 print('The MODEL_TYPE is invalid.')
                 exit()
        # Build the channel codec, modulator, and channel physical model
        if MODEL_TYPE in ['JPEG', 'J2K', 'BPG']:
            self.tradicom = TradiCom(com_params['tradicom'])
        elif MODEL_TYPE == 'JSCC' or MODEL_TYPE == 'AE':
            self.channel_phy = Channels(self.device)
        elif MODEL_TYPE == 'ORIGIN':
            pass
        else:
            print('The MODEL_TYPE is invalid.')
            exit()
        
        self.img_last = None

        # if PATH_CH_MODEL is not None:
        #     self.ch_manager = ChannelCodec(1024)
        #     self.ch_manager.to(self.device)
        #     channel_weights = torch.load(PATH_CH_MODEL, map_location=self.device)
        #     self.ch_manager.load_state_dict(channel_weights['model'], strict=False)
        #     self.ch_manager.eval()
        #     self.channel_phy = Channels(self.device)
        # ====================================================================>

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def sensors(self):
                return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': -1.5, 'y': 0.0, 'z':2.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 900, 'height': 256, 'fov': 100,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'bev'
                    },    
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
        
        result = {
                'rgb': rgb,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                'bev': bev
                }
        
        pos = self._get_position(result)
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value
        
        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)
        
        # <=========================
        # info_show(rgb, 'rgb', False) # (256, 900, 3) [0~255]
        # info_show(bev, 'bev', False) # (512, 512, 3)
        # info_show(gps, 'gps') # (2,)

        # info_show(speed, 'speed') # numpy.float64
        # info_show(compass, 'compass') # numpy.float64
        # info_show(pos, 'pos') # (2,)
        # info_show(next_cmd, 'next_cmd') # RoadOption
        # info_show(result['target_point'], 'target_point') # tuple len=2

        # info_show(input_data['rgb'][1], 'input_data', True) # (256, 900, 4) [0~255]
        # =========================>

        return result
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        
        if self.step < self.config.seq_len:
            rgb = self._im_transform(tick_data['rgb']).unsqueeze(0)

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            
            return control

        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
        command = tick_data['next_command']
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
        speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
        speed = speed / 12
        # Add an extra dimension for AI input
        # info_show(tick_data['rgb'][0,0,0], 'rgb')
        rgb = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)
        # info_show(rgb, 'rgb')
        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
                                        torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)
        state = torch.cat([speed, target_point, cmd_one_hot], 1)
        
        # <=========================
        with torch.no_grad():
            if MODEL_TYPE == 'JSCC' or MODEL_TYPE == 'AE':
                rgb, tick_data = self.__2nd_process(tick_data, state, rgb)
            elif MODEL_TYPE in ['JPEG', 'J2K', 'BPG']:
                rgb, tick_data = self.__simple_process(tick_data, quality=QUALITY)
        # =========================>
        pred= self.net(rgb, state, target_point)

        steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, tick_data['next_command'], gt_velocity, target_point)

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity, target_point)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        self.pid_metadata = metadata_traj
        control = carla.VehicleControl()

        if self.status == 0:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'traj'
            control.steer = np.clip(self.alpha*steer_ctrl + (1-self.alpha)*steer_traj, -1, 1)
            control.throttle = np.clip(self.alpha*throttle_ctrl + (1-self.alpha)*throttle_traj, 0, 0.75)
            control.brake = np.clip(self.alpha*brake_ctrl + (1-self.alpha)*brake_traj, 0, 1)
        else:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'ctrl'
            control.steer = np.clip(self.alpha*steer_traj + (1-self.alpha)*steer_ctrl, -1, 1)
            control.throttle = np.clip(self.alpha*throttle_traj + (1-self.alpha)*throttle_ctrl, 0, 0.75)
            control.brake = np.clip(self.alpha*brake_traj + (1-self.alpha)*brake_ctrl, 0, 1)


        self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
        self.pid_metadata['brake_traj'] = float(brake_traj)

        if control.brake > 0.5:
            control.throttle = float(0)

        if len(self.last_steers) >= 20:
            self.last_steers.popleft()
        self.last_steers.append(abs(float(control.steer)))
        #chech whether ego is turning
        # num of steers larger than 0.1
        num = 0
        for s in self.last_steers:
            if s > 0.10:
                num += 1
        if num > 10:
            self.status = 1
            self.steer_step += 1

        else:
            self.status = 0

        self.pid_metadata['status'] = self.status

        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data)
        return control

    def save(self, tick_data):
        frame = self.step // 10

        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.net
        if PATH_VAE_MODEL is not None:
            del self.codec
        torch.cuda.empty_cache()
        if USE_WANDB == 'True':
            wandb.log({'Time': time.time()})
    
    def __2nd_process(self, tick_data, state, rgb_tcp):
        # Change the channel from H*W*C to C*H*W
        rgb = torch.tensor(tick_data['rgb']).to(self.device, dtype=torch.float32).permute(2, 0, 1)
        # Change range to [0, 1]
        rgb = rgb.unsqueeze(0) / 255
        rgb = self.norm_manager.norm(rgb)
        # info_show(rgb, '2nd_rgb')
        
        # VAE Encoding
        img_mu, img_logvar = self.codec.encode(rgb)
        # img_z = reparameterize(img_mu, img_logvar)
        img_mu = power_norm(img_mu, com_params['jscc']['power'])
        # Adding noise
        if com_params['jscc']['noise_type'] == 'AWGN':
            img_mu_rec = self.channel_phy.awgn(img_mu, com_params['jscc']['snr_db'], com_params['jscc']['power'])
        # elif com_params['jscc']['noise_type'] == 'Rayleigh' or com_params['jscc']['noise_type'] == 'Rician':
        #     img_mu_rec = self.channel_phy.fading(img_mu, com_params['jscc']['snr_db'], K=com_params['jscc']['k_ratio'])
        else:
            print('No noise is added!')
            img_mu_rec = img_mu
        
        batch_img_rec = self.codec.decode(img_mu_rec)
        # For saving
        tick_data['rgb'] = self.norm_manager.image_2_rawImage(batch_img_rec)
        # info_show(rgb_recon, '2nd_rgb_recon')
        rgb_recon = self.norm_manager.realBatch_2_tcpBatch(batch_img_rec)
        
        return rgb_recon, tick_data
    
    def __simple_process(self, tick_data: Dict, quality: int = 1):
        # tick_data['rgb'], rgb, and self.img_last are similar
        rgb = tick_data['rgb']
        
        # source encoding
        bits_tensor, bits_length = self.codec.encode(rgb, quality)
        # channel coding -> modulation -> channel -> demodulation -> channel decoding
        bits_tensor = self.tradicom.padding_bits(bits_tensor)
        if self.img_last is None:
            bits_tensor = self.tradicom.e2e_com(bits_tensor, 30)
        else:
            bits_tensor = self.tradicom.e2e_com(bits_tensor, com_params['tradicom']['snr_db'])
        # source decoding
        rgb_recon = self.codec.decode(bits_tensor, bits_length)
        # initialize the last image
        if self.img_last is None:
            self.img_last = copy.deepcopy(rgb_recon)
        # Use last image
        if rgb_recon is None:
            rgb_recon = self.img_last
        elif rgb_recon.shape != self.img_last.shape:
            rgb_recon = self.img_last
        else:
            pass
        
        tick_data['rgb'] = rgb_recon
        
        try:
            rgb_recon_trans = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)
        except TypeError:
            print('The rgb_recon is invalid!')
            rgb_recon = self.img_last
            tick_data['rgb'] = rgb_recon
            rgb_recon_trans = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)

        # Update the last image
        self.img_last = copy.deepcopy(rgb_recon)

        return rgb_recon_trans, tick_data
        
        
        
    
