import argparse
import os
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import traceback
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import write_json
import random


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def load_ckpt(model, ckpt_path):
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
    model.load_state_dict(state_dict_new)
    return model

def load_ckpt_efnet(model, ckpt_path):
    state_dict = torch.load(ckpt_path)
    #config = ckpt['config']
    #write_json(config.config, ckpt_path+'config.json')
    model.load_state_dict(state_dict['params'])
    return model

def load_model(checkpoint=None, config=None):
    """
    negative voxel indicates a model trained on negative voxels -.-
    """
    resume = checkpoint is not None
    if resume:
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
    try:
        model_info['num_bins'] = config['arch']['args']['unet_kwargs']['num_bins']
    except KeyError:
        model_info['num_bins'] = config['arch']['args']['num_bins']
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    if resume:
        model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.train()
    return model

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = config.init_obj('valid_data_loader', module_data)

    # build model architecture, then print to console
    # the event reconstruction model
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    try:
       model.model_evt_rec = load_ckpt(model.model_evt_rec, config['resume_ckpt_evt_rec'])
       print('evt_rec load success')
    except:
       traceback.print_exc()
       pass

    try:
       model.efnet = load_ckpt_efnet(model.efnet, config['resume_ckpt_efnet'])
       print('efnet load success')
    except:
       traceback.print_exc()
       pass

    try:
       model = load_ckpt(model, config['resume_ckpt_full'])
       print('full model load success')
    except:
       traceback.print_exc()
       pass

    #try:
    #   checkpoint = torch.load(config['resume_ckpt_full'])
    #   model = load_model(checkpoint,config)
    #except:
    #   pass
    # init loss classes
    loss_ftns = [getattr(module_loss, loss)(**kwargs) for loss, kwargs in config['loss_ftns'].items()]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # in deblur we dont train model_evt_rec
    trainable_params = []
    trainable_params += list(filter(lambda p: p.requires_grad, model.efnet.parameters()))
    #trainable_params += list(filter(lambda p: p.requires_grad, model.model_evt_rec.parameters()))
    trainable_params += list(filter(lambda p: p.requires_grad, model.forward_rnn_encoder.parameters()))
    trainable_params += list(filter(lambda p: p.requires_grad, model.backward_rnn_encoder.parameters()))
    trainable_params += list(filter(lambda p: p.requires_grad, model.CTCA_evt_rec_evt_deblur.parameters()))
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)


    trainer = Trainer(model, loss_ftns, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


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

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--rmb', '--reset_monitor_best'], type=bool, target='trainer;reset_monitor_best'),
        CustomArgs(['--vo', '--valid_only'], type=bool, target='trainer;valid_only')
    ]
    config = ConfigParser.from_args(args, options)

    if args.parse_args().limited_memory:
        # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    main(config)
