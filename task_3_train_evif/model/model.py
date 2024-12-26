import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# local modules

from .model_util import CropParameters, recursive_clone
from .base.base_model import BaseModel

from .unet import UNetFlow, WNet, UNetFlowNoRecur, UNetRecurrent, UNet, UNetFlow_encoder_backward, UNetFlow_encoder_forward, UNetFlowNoRecur_vis, CorrFeatSplittingModules, DConvModules, TempCrossAttnModules, LSTMEncoder, CTCA_stages
from .submodules import ResidualBlock, ConvGRU, ConvLayer
from utils.color_utils import merge_channels_into_color_image

from .legacy import FireNet_legacy
from .ref_EFNet_arch import EFNet
from .evif_arch import ConvEncoder, ConvDecoder, DepthwiseSeparableConv, FusionTransformers, MI_Minimization_reg, MI_Maximization_reg

def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)


class ColorNet(BaseModel):
    """
    Split the input events into RGBW channels and feed them to an existing
    recurrent model with states.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.channels = {'R': [slice(0, None, 2), slice(0, None, 2)],
                         'G': [slice(0, None, 2), slice(1, None, 2)],
                         'B': [slice(1, None, 2), slice(1, None, 2)],
                         'W': [slice(1, None, 2), slice(0, None, 2)],
                         'grayscale': [slice(None), slice(None)]}
        self.prev_states = {k: self.model.states for k in self.channels}

    def reset_states(self):
        self.model.reset_states()

    @property
    def num_encoders(self):
        return self.model.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with RGB image taking values in [0, 1], and
                 displacement within event_tensor.
        """
        height, width = event_tensor.shape[-2:]
        crop_halfres = CropParameters(int(width / 2), int(height / 2), self.model.num_encoders)
        crop_fullres = CropParameters(width, height, self.model.num_encoders)
        color_events = {}
        reconstructions_for_each_channel = {}
        for channel, s in self.channels.items():
            color_events = event_tensor[:, :, s[0], s[1]]
            if channel == 'grayscale':
                color_events = crop_fullres.pad(color_events)
            else:
                color_events = crop_halfres.pad(color_events)
            self.model.states = self.prev_states[channel]
            img = self.model(color_events)['image']
            self.prev_states[channel] = self.model.states
            if channel == 'grayscale':
                img = crop_fullres.crop(img)
            else:
                img = crop_halfres.crop(img)
            img = img[0, 0, ...].cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            reconstructions_for_each_channel[channel] = img
        image_bgr = merge_channels_into_color_image(reconstructions_for_each_channel)  # H x W x 3
        return {'image': image_bgr}


class WFlowNet(BaseModel):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.wnet = WNet(unet_kwargs)

    def reset_states(self):
        self.wnet.states = [None] * self.wnet.num_encoders

    @property
    def states(self):
        return copy_states(self.wnet.states)

    @states.setter
    def states(self, states):
        self.wnet.states = states

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.wnet.forward(event_tensor)
        return output_dict


class FlowNet(BaseModel):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        ks_bak = unet_kwargs['kernel_size']
        unet_kwargs['kernel_size'] = 5
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetflow = UNetFlow(unet_kwargs)
        unet_kwargs['kernel_size'] = ks_bak

    @property
    def states(self):
        return copy_states(self.unetflow.states)

    @states.setter
    def states(self, states):
        self.unetflow.states = states

    def reset_states(self):
        self.unetflow.states = [None] * self.unetflow.num_encoders

    def forward(self, event_tensor, state1, state2, state3):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetflow.forward(event_tensor,state1, state2, state3)
        return output_dict


class FlowNetNoRecur(BaseModel):
    """
    UNet-like architecture without recurrent units
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetflow = UNetFlowNoRecur(unet_kwargs)

    def reset_states(self):
        pass

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetflow.forward(event_tensor)
        return output_dict


class E2VIDRecurrent(BaseModel):
    """
    Compatible with E2VID_lightweight
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrent(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetrecurrent.forward(event_tensor)
        return output_dict


class EVFlowNet(BaseModel):
    """
    Model from the paper: "EV-FlowNet: Self-Supervised Optical Flow for Event-based Cameras", Zhu et al. 2018.
    Pytorch adaptation of https://github.com/daniilidis-group/EV-FlowNet/blob/master/src/model.py (may differ slightly)
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        # put 'hardcoded' EVFlowNet parameters here
        EVFlowNet_kwargs = {
            'base_num_channels': 32, # written as '64' in EVFlowNet tf code
            'num_encoders': 4,
            'num_residual_blocks': 2,  # transition
            'num_output_channels': 2,  # (x, y) displacement
            'skip_type': 'concat',
            'norm': None,
            'use_upsample_conv': True,
            'kernel_size': 3,
            'channel_multiplier': 2
            }
        unet_kwargs.update(EVFlowNet_kwargs)

        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unet = UNet(unet_kwargs)

    def reset_states(self):
        pass

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with N x 2 X H X W (x, y) displacement within event_tensor.
        """
        flow = self.unet.forward(event_tensor)
        # to make compatible with our training/inference code that expects an image, make a dummy image.
        return {'flow': flow, 'image': 0 * flow[..., 0:1, :, :]}


class FireNet(BaseModel):
    """
    Refactored version of model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
    The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
    However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs={}):
        super().__init__()
        if unet_kwargs:  # legacy compatibility - modern config should not have unet_kwargs
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states()

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H x W image
        """
        x = self.head(x)
        x = self.G1(x, self._states[0])
        self._states[0] = x
        x = self.R1(x)
        x = self.G2(x, self._states[1])
        self._states[1] = x
        x = self.R2(x)
        return {'image': self.pred(x)}


class FlowNet_deblur_evt(BaseModel):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    just have encoders, it have two encoders (forward and backward)
    forming a bi-directional event representation
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        num_bins_bak = unet_kwargs['num_bins']
        unet_kwargs['num_bins'] = 2 # each time takes 2 bins as input
        self.evt_encoder_backward = UNetFlow_encoder_backward(unet_kwargs)
        self.evt_encoder_forward  = UNetFlow_encoder_forward(unet_kwargs)
        unet_kwargs['num_bins'] = num_bins_bak

    @property
    def states(self):
        return (copy_states(self.evt_encoder_backward.states), copy_states(self.evt_encoder_forward.states))

    @states.setter
    def states(self, states_b, states_f):
        self.evt_encoder_backward.states = states_b
        self.evt_encoder_forward.states = states_f

    def reset_states(self):
        self.evt_encoder_backward.states = [None] * self.evt_encoder_backward.num_encoders
        self.evt_encoder_forward.states = [None] * self.evt_encoder_forward.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        num_bins are 2*K channels that has
        [T0_pos, T1_pos, T2_pos, ...TK_pos, T0_neg, ...TK_neg}
        each time we take [Ti_pos, Ti_neg] as input
        perform a backward prop first
        then perform a forward prop
        :return: output dict with evt features"""
        nc = event_tensor.shape[1]
        nt = nc // 2
        pos_evt_tensor = event_tensor[:,0:nt,:,:]
        neg_evt_tensor = event_tensor[:,nt:,:,:]
        
        # performing backward prop first
        backward_feat = [None] * nt
        for i in range(nt-1, -1, -1):
            pos_bin = pos_evt_tensor[:,i:i+1,:,:]
            neg_bin = neg_evt_tensor[:,i:i+1,:,:]
            evt_input = torch.cat((pos_bin,neg_bin),1)
            output_dict = self.evt_encoder_backward.forward(evt_input)
            backward_feat[i] = output_dict['evt_feat']

        forward_feat = [None] * nt
        for i in range(0, nt, 1):
            pos_bin = pos_evt_tensor[:,i:i+1,:,:]
            neg_bin = neg_evt_tensor[:,i:i+1,:,:]
            evt_input = torch.cat((pos_bin,neg_bin),1)
            output_dict = self.evt_encoder_forward.forward(evt_input, backward_feat[i])
            forward_feat[i] = output_dict['evt_feat']


        # take the last timestamp's forward feat as output
        output_dict = {'evt_feat':forward_feat[-1]}

        return output_dict

class FlowNet_deblur_vis(BaseModel):
    """
    UNet-like architecture without recurrent units
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = 1 # just take one channel blur infrared input
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        num_bins_bak = unet_kwargs['num_bins']
        unet_kwargs['num_bins'] = 1 # each time takes 1 bins as input
        self.unetflow = UNetFlowNoRecur_vis(unet_kwargs)
        unet_kwargs['num_bins'] = num_bins_bak

    def reset_states(self):
        pass

    def forward(self, blur, evt_rec_feats, evt_deblur_feats):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetflow.forward(blur, evt_rec_feats, evt_deblur_feats)
        return output_dict


class Full_fusion_model(BaseModel):
    """
    UNet-like architecture without recurrent units
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.efnet = EFNet(**unet_kwargs['EFNet_args'])
        del unet_kwargs['EFNet_args']
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        # since evt rec net uses sum, not concat
        skip_type_bak = unet_kwargs['skip_type']
        unet_kwargs['skip_type'] = 'sum'
        self.model_evt_rec = FlowNet(unet_kwargs)
        unet_kwargs['skip_type'] = skip_type_bak

        self.forward_rnn_encoder     = LSTMEncoder(self.model_evt_rec,self.efnet)
        self.backward_rnn_encoder    = LSTMEncoder(self.model_evt_rec,self.efnet)
        self.CTCA_evt_rec_evt_deblur = CTCA_stages(self.model_evt_rec,self.efnet)

        self.fusion_enc_vis = ConvEncoder(input_nc=1)
        self.fusion_enc_inf = ConvEncoder(input_nc=1)
        self.fusion_dec     = ConvDecoder(output_nc=1)
        self.align_dws_conv_inf = nn.ModuleList()

        for i in range(self.fusion_enc_inf.num_blocks+1):
            pretask_nc = self.forward_rnn_encoder.efnet_num_channels[i]
            encoder_nc = self.fusion_enc_inf.channels[i]
            total_nc = pretask_nc + encoder_nc 
            self.align_dws_conv_inf.append(DepthwiseSeparableConv(total_nc, self.fusion_enc_inf.channels[i]))
        
        self.align_dws_conv_vis = nn.ModuleList()
        for i in range(self.fusion_enc_vis.num_blocks+1):
            pretask_nc = self.model_evt_rec.unetflow.encoder_input_sizes[i]
            encoder_nc = self.fusion_enc_vis.channels[i]
            total_nc = pretask_nc + encoder_nc 
            self.align_dws_conv_vis.append(DepthwiseSeparableConv(total_nc, self.fusion_enc_vis.channels[i]))

        self.fusion_transformers = nn.ModuleList()
        window_sizes = [8, 4, 2]
        for i in range(self.fusion_enc_vis.num_blocks+1):
            dim = self.fusion_enc_vis.channels[i]
            self.fusion_transformers.append(FusionTransformers(dim, img_size=256, patch_size=16, embed_dim=dim, window_size=window_sizes[i])) 
        
        self.mi_minimizers = nn.ModuleList()
        input_sizes = [128, 64, 32]
        latent_sizes = [6, 6, 6]
        for i in range(self.fusion_enc_vis.num_blocks+1):
            dim = self.fusion_enc_vis.channels[i]
            self.mi_minimizers.append(MI_Minimization_reg(dim, dim, latent_sizes[i], input_size=input_sizes[i]))

        self.mi_maximizers_inf = nn.ModuleList()
        self.mi_maximizers_vis = nn.ModuleList()
        input_sizes = [256, 128, 64]
        latent_sizes = [256,128,64]
        for i in range(self.fusion_enc_vis.num_blocks+1):
            if i > 0:
                continue
            dim = self.fusion_enc_vis.channels[i]
            sobel = False
            self.mi_maximizers_inf.append(MI_Maximization_reg(dim, dim, latent_sizes[i], input_size=input_sizes[i], sobel=sobel))
            self.mi_maximizers_vis.append(MI_Maximization_reg(dim, dim, latent_sizes[i], input_size=input_sizes[i], sobel=sobel))


    def reset_states(self):
        self.model_evt_rec.reset_states()
        pass

    
    def forward(self, blur, deblur_evt_bin, rec_history_evt_bins, rec_prev_history_evt_bins):
        """
        blur: blurry infrared
        deblur_evt_bin: evt voxel corresponding to the exposure time of the blurry infrared
        rec_history_evt_bins: several (set to 6 in dataset) event voxels corresponding to the current exposure  windows, shape [N, K, C, H, W], K is history size
        rec_prev_history_evt_bins: several (set to 20 in dataset) history event voxels corresponding to the history reconstruction windows, shape [N, K, C, H, W], K is history size
        """
        # self.reset_states() # this is run in trainer.py
        # first we calculate recurrent event rec features
        with torch.no_grad():
            states_1 = None 
            states_2 = None 
            states_3 = None 
            all_time_evt_rec_feats = [] # a list of T small lists, each list contains K feats
            # we shall first do few rounds of warm up using the prev evt voxel
            for i in range(rec_prev_history_evt_bins.shape[1]):
                input_bin = rec_prev_history_evt_bins[:,i,:,:,:]
                output_dict = self.model_evt_rec.forward(input_bin, states_1, states_2, states_3)
                states_1 = output_dict['state1']
                states_2 = output_dict['state2']
                states_3 = output_dict['state3']
            # we need the last feat in prev_history
            evt_rec_feats_i = output_dict['evt_feat']
            evt_rec_feats_i = [f.detach() for f in evt_rec_feats_i]
            all_time_evt_rec_feats.append(evt_rec_feats_i)
            # then we get the evt rec features corresponding to the blurry frame's exposure window
            evt_rec_mid_T_rec_res = None
            evt_rec_mid_T_feat = None
            for i in range(rec_history_evt_bins.shape[1]):
                input_bin = rec_history_evt_bins[:,i,:,:,:]
                output_dict = self.model_evt_rec.forward(input_bin, states_1, states_2, states_3)
                # get evt rec feature
                evt_rec_feats_i = output_dict['evt_feat']
                states_1 = output_dict['state1']
                states_2 = output_dict['state2']
                states_3 = output_dict['state3']
                # we just not update evt rec net
                evt_rec_feats_i = [f.detach() for f in evt_rec_feats_i]
                all_time_evt_rec_feats.append(evt_rec_feats_i)
                # save reconstructed vis image corresponding to the mid time of the blurry infrared image
                if i ==( (rec_history_evt_bins.shape[1] // 2) - 1):
                    evt_rec_mid_T_rec_res = output_dict['image']
                    evt_rec_mid_T_dec_feat = output_dict['decoder_feats']

            stage_T_feats = [] # a list of K stage feats, each shape [N, T, C, H, W]
            for i in range(len(all_time_evt_rec_feats[0])):
                stage_i_T_feats = []
                for j in range(len(all_time_evt_rec_feats)):
                    stage_i_T_feats.append(all_time_evt_rec_feats[j][i])
                stage_i_T_feats = torch.stack(stage_i_T_feats,1)
                stage_T_feats.append(stage_i_T_feats)



            # next process stage_T_feats with two RNNs
            RNN_f = []
            for i in range(len(stage_T_feats)):
                fwd_RNN_f = self.forward_rnn_encoder(stage_T_feats[i], i)
                bwd_RNN_f = self.backward_rnn_encoder(stage_T_feats[i], i, reverse=True)
                stage_RNN_f = torch.cat((fwd_RNN_f, bwd_RNN_f), 1)
                RNN_f.append(stage_RNN_f)

            # next get efnet event features
            ef_evt_f = self.efnet.forward_evt(deblur_evt_bin)

            # next fuse RNN_f with ef_evt_f
            merged_evt_feats = []
            for i in range(len(stage_T_feats)):
                fused = self.CTCA_evt_rec_evt_deblur(RNN_f[i], ef_evt_f[i], i)
                merged_evt_feats.append(fused)

            # next we run rest stage of efnet

            # first run prestage of efnet
            if blur.shape[1] == 1:
                blur = torch.cat((blur,blur,blur),1)

            ef_f = self.efnet.forward_before_enc(blur, merged_evt_feats)

            # then merge stage feats
            for i in range(self.forward_rnn_encoder.num_stages):
                # get efnet feat
                ef_f = self.efnet.forward_enc_stage(ef_f, i)

            # forward the rest part of efnet 
            [out_1, out_2, inf_dec_feats] = self.efnet.forward_rest(ef_f)
            # next get deblur result
            output_dict = {'sharp_inf_img': out_2, 'inf_dec_feats': inf_dec_feats, 'evt_rec_img': evt_rec_mid_T_rec_res, 'evt_rec_dec_feats': evt_rec_mid_T_dec_feat}

        # Now we have finished the first two tasks, start fusion

        # prepare everything 
        img_inf = output_dict['sharp_inf_img']
        img_vis = output_dict['evt_rec_img']
        pre_task_feat_inf = output_dict['inf_dec_feats']
        pre_task_feat_vis = output_dict['evt_rec_dec_feats']
        # go through encodrs
        if img_inf.shape[1] == 3:
            img_inf = torch.mean(img_inf, 1, keepdim=True)
            output_dict['sharp_inf_img'] = img_inf
        fusion_enc_feat_inf = self.fusion_enc_inf.get_resblock_features(img_inf)

        if img_vis.shape[1] == 3:
            img_vis = torch.mean(img_vis, 1, keepdim=True)
            output_dict['evt_rec_img'] = img_vis
        fusion_enc_feat_vis = self.fusion_enc_vis.get_resblock_features(img_vis)

        # cat pre task feature with fusion enc feature
        cat_feat_inf = []
        cat_feat_vis = []
        for i in range(len(fusion_enc_feat_inf)):
            cat_feat_inf.append(torch.cat((pre_task_feat_inf[len(fusion_enc_feat_inf)-1-i],fusion_enc_feat_inf[i]),1))
            cat_feat_vis.append(torch.cat((pre_task_feat_vis[len(fusion_enc_feat_inf)-1-i],fusion_enc_feat_vis[i]),1))

        # align channel_num with depth-wise separable conv
        evif_feat_inf = []
        evif_feat_vis = []
        for i in range(len(cat_feat_inf)):
            evif_feat_inf.append(self.align_dws_conv_inf[i](cat_feat_inf[i]))
        for i in range(len(cat_feat_vis)):
            evif_feat_vis.append(self.align_dws_conv_vis[i](cat_feat_vis[i]))


        # use transformer layers to fuse the features
        f_inf_filtered = []
        f_vis_filtered = []
        
        for i in range(len(self.fusion_transformers)):
            transformers = self.fusion_transformers[i]
            f_inf, f_vis = transformers(evif_feat_inf[i], evif_feat_vis[i])
            f_inf_filtered.append(f_inf)
            f_vis_filtered.append(f_vis)


        # calculate MI minimization loss
        mi_loss = 0
        inf_feat_attn = []
        vis_feat_attn = []
        for i in range(len(self.mi_minimizers)):
            mi_minimizer = self.mi_minimizers[i]
            mi_out_dict = mi_minimizer(f_inf_filtered[i], f_vis_filtered[i])
            mi_loss += mi_out_dict['latent_loss']
            inf_feat_attn.append(mi_out_dict['inf_feat_attn'])
            vis_feat_attn.append(mi_out_dict['vis_feat_attn'])
        output_dict['mi_minimization_loss'] = mi_loss

        f_fused = []
        for i in range(len(self.fusion_transformers)):
            transformers = self.fusion_transformers[i]
            f_1 = f_inf_filtered[i] * inf_feat_attn[i]
            f_2 = f_vis_filtered[i] * vis_feat_attn[i]
            f = transformers.fuse(f_1, f_2)      
            f_fused.append(f)

        # get fused image
        fused_img = self.fusion_dec(f_fused)

        output_dict['fused_img'] = fused_img

        fusion_enc_feat_inf_fus = self.fusion_enc_inf.get_resblock_features(fused_img)
        fusion_enc_feat_vis_fus = self.fusion_enc_vis.get_resblock_features(fused_img)

        # calculate MI maximization InfoNCE loss
        ma_loss = 0
        for i in range(len(self.mi_maximizers_inf)):
            ma_loss_i = 0
            idx = i
            mi_maximizer_inf = self.mi_maximizers_inf[i]
            ma_out_dict_inf = mi_maximizer_inf(fusion_enc_feat_inf[idx], fusion_enc_feat_inf_fus[idx])
            ma_loss_inf = ma_out_dict_inf['latent_loss']
            ma_dist_inf = ma_out_dict_inf['dist']
            mi_maximizer_vis = self.mi_maximizers_vis[i]
            ma_out_dict_vis = mi_maximizer_vis(fusion_enc_feat_vis[idx], fusion_enc_feat_vis_fus[idx])
            ma_loss_vis = ma_out_dict_vis['latent_loss']
            ma_dist_vis = ma_out_dict_vis['dist']
            mask = ma_dist_inf < ma_dist_vis
            mask = mask.squeeze().float()
            ma_loss_i = (mask * ma_loss_vis + (1-mask) * ma_loss_inf).sum()
            ma_loss_i = ma_loss_i / float(ma_loss_inf.shape[0])
            ma_loss = ma_loss + ma_loss_i

        output_dict['mi_maximization_loss'] = ma_loss

        return output_dict







