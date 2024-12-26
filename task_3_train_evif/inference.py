import argparse
import torch
import collections
import numpy as np
from os.path import join
import data_loader.data_loaders as module_data
import model.model as module_arch
import os
import cv2
from tqdm import tqdm

from utils.util import ensure_dir, flow2bgr_np
from data_loader.data_loaders import InferenceDataLoader
from model.model import ColorNet
from utils.util import CropParameters, get_height_width, torch2cv2, \
                       append_timestamp, setup_output_folder, ensure_dir
from utils.timers import CudaTimer
from utils.henri_compatible import make_henri_compatible

from parse_config import ConfigParser

from model.loss import psnr_loss

model_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ckpt(model, ckpt_path, strict=True):
    ckpt = torch.load(ckpt_path)
    #config = ckpt['config']
    #write_json(config.config, ckpt_path+'config.json')
    state_dict = ckpt['state_dict']
    state_dict_new = {}
    for k in state_dict.keys():
        v = state_dict[k]
        if 'module.' in k:
            state_dict_new[k[7:]] = v
        else:
            state_dict_new[k] = v
    model.load_state_dict(state_dict_new, strict=strict)
    return model

def legacy_compatibility(args, checkpoint):
    assert not (args.e2vid and args.firenet_legacy)
    if args.e2vid:
        args.legacy_norm = True
        final_activation = 'sigmoid'
    elif args.firenet_legacy:
        args.legacy_norm = True
        final_activation = ''
    else:
        return args, checkpoint
    # Make compatible with Henri saved models
    if not isinstance(checkpoint.get('config', None), ConfigParser) or args.e2vid or args.firenet_legacy:
        checkpoint = make_henri_compatible(checkpoint, final_activation)
    if args.firenet_legacy:
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'
    return args, checkpoint


def load_model(checkpoint):
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    logger.info(model)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    if args.color:
        model = ColorNet(model)
    for param in model.parameters():
        param.requires_grad = False

    return model


def main(args, config):
    data_loader = config.init_obj('valid_data_loader', module_data)

    height, width = get_height_width(data_loader)

    model_info['input_shape'] = height, width

    model = config.init_obj('arch', module_arch)
    model = load_ckpt(model, config['resume_ckpt_full'], strict=True)
    model = model.to(device)
    model.eval()
    ensure_dir(args.output_folder)
    
    model.reset_states()
    ploss = psnr_loss()
    mean_psnr = 0.0
    count = 0
    for i in range(0, len(data_loader), args.skip_frames):
        item = data_loader.dataset[i]
        model.reset_states()
        blur           = item['blurry_frame'].float().to(device).unsqueeze(0)
        blurry_events  = item['blurry_events'].float().to(device).unsqueeze(0)
        history_events = item['history_events'].float().to(device).unsqueeze(0)
        prev_history_events = item['prev_history_events'].float().to(device).unsqueeze(0)
        image          = item['sharp_frame'].float().to(device).unsqueeze(0)

        # normalize input and gt
        min_vals = blur.view(blur.shape[0],-1).min(dim=1, keepdim=True).values
        min_vals = min_vals.unsqueeze(-1).unsqueeze(-1)
        max_vals = blur.view(blur.shape[0],-1).max(dim=1, keepdim=True).values
        max_vals = max_vals.unsqueeze(-1).unsqueeze(-1)
        blur_norm = (blur - min_vals) / (max_vals - min_vals + 1e-6)
        gt_norm = (image - min_vals) / (max_vals - min_vals + 1e-6)

        with CudaTimer('Inference'):
            with torch.no_grad():
                output = model(blur_norm, blurry_events, history_events, prev_history_events)
        inf_image = output['sharp_inf_img']
        inf_image = torch2cv2(inf_image)
        vis_image = output['evt_rec_img']
        vis_image = torch2cv2(vis_image)
        fuse_image = output['fused_img']
        fuse_image = torch2cv2(fuse_image)
        fname = 'frame_{:010d}_02_inf.png'.format(i)
        cv2.imwrite(join(args.output_folder, fname), inf_image)
        fname = 'frame_{:010d}_01_vis.png'.format(i)
        cv2.imwrite(join(args.output_folder, fname), vis_image)
        fname = 'frame_{:010d}_03_fuse.png'.format(i)
        cv2.imwrite(join(args.output_folder, fname), fuse_image)
        count += 1
        print(join(args.output_folder, fname))



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--limited_memory', default=False, action='store_true',
                      help='prevent "too many open files" error by setting pytorch multiprocessing to "file_system".')
    args.add_argument('--output_folder', default="/tmp/output", type=str,
                        help='where to save outputs to')
    args.add_argument('--skip_frames', default=1, type=int,
                        help='test one frame every [skip_frames] frames in the test set')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--rmb', '--reset_monitor_best'], type=bool, target='trainer;reset_monitor_best'),
        CustomArgs(['--vo', '--valid_only'], type=bool, target='trainer;valid_only')
    ]


    if args.parse_args().limited_memory:
        # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
    
    config = ConfigParser.from_args(args, options)

    args = args.parse_args()
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device


    main(args,config)
