import argparse
import os
from mmcv.runner import get_dist_info
import torch.distributed as dist
from os.path import join as pjoin
import yaml

class TrainCompOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        self.parser.add_argument('--lr', type=float, default=1e-4)
        #If you want the generated hands to match body movements, train using these parameter.
        self.parser.add_argument('--bodystyleloss_weight', type=float, default=0.8)
        self.parser.add_argument('--textstyleloss_weight', type=float, default=0.2)

        #If you want to enhance text control over hands, train using these parameter.
        # self.parser.add_argument('--bodystyleloss_weight', type=float, default=0.2)
        # self.parser.add_argument('--textstyleloss_weight', type=float, default=0.8)
        
        self.parser.add_argument('--config', default='eval.yaml')
        
        

class OptParse():
    def __init__(self):
        train_options = TrainCompOptions()
        self.opt = train_options.parser.parse_args()
        self.config = load_config(self.opt.config)
        
        self.opt.mode = self.config['mode']
        self.opt.name = self.config['name']
        self.opt.device = self.config['device']
        self.opt.seed = self.config['seed']
        
        self.opt.d_model = self.config['model']['d_model']
        self.opt.n_dec_layers = self.config['model']['n_dec_layers']
        self.opt.n_head = self.config['model']['n_head']
        self.opt.d_k = self.config['model']['d_k']
        self.opt.d_v = self.config['model']['d_v']
        self.opt.timesteps = self.config['model']['timesteps']
        
        self.opt.window = self.config['dataset']['window']
        self.opt.batch_size = self.config['dataset']['batch_size']
        self.opt.frame_rate = self.config['dataset']['frame_rate']
        self.opt.have_hand = self.config['dataset']['have_hand']
        self.opt.output_mode = self.config['dataset']['output_mode']
        self.opt.need_fix_initface = self.config['dataset']['need_fix_initface']
        
        self.opt.datasets_to_process = self.config['datasets_to_process']
        self.opt.checkpoints_dir = self.config['checkpoints_dir']
        self.opt.data_root_folder = self.config['data_root_folder']
        self.opt.resume_ckpt_path = self.config['resume_ckpt_path']
        self.opt.rewake = self.config['rewake']
        self.opt.save_top_k = self.config['save_top_k']
        self.opt.save_last = self.config['save_last']
        self.opt.save_frequency = self.config['save_frequency']
        self.opt.num_epochs = self.config['num_epochs']
        self.opt.style_epochs = self.config['style_epochs']
        self.opt.print_every = self.config['print_every']
        self.opt.log_every = self.config['log_every']
        self.opt.vis_every = self.config['vis_every']
        self.opt.gen_times = self.config['gen_times']
        self.opt.validate_times = self.config['validate_times']
        self.opt.validate_batch_size = self.config['validate_batch_size']
        self.opt.costume_promot = self.config['costume_promot']
        
        rank, world_size = get_dist_info()
        self.opt.save_root = pjoin(self.opt.checkpoints_dir, self.opt.name)
        self.opt.model_dir = pjoin(self.opt.save_root, 'model')
        self.opt.checkpoint_dir = pjoin(self.opt.save_root, 'checkpoints')
        self.opt.log_dir = pjoin(self.opt.save_root, 'log')
        self.opt.vis_dir = pjoin(self.opt.save_root, 'vis_res')
        self.opt.evl_dir = pjoin(self.opt.save_root, 'evl_res')

                
def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(os.path.join('./config',path), "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        
    return cfg