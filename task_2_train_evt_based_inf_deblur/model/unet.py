import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from .submodules import \
    ConvLayer, UpsampleConvLayer, TransposedConvLayer, \
    RecurrentConvLayer, ResidualBlock, ConvLSTM, \
    ConvGRU, RecurrentResidualLayer

from .model_util import *
from .ref_EFNet_arch import UNetConvBlock, UNetUpBlock
from .dcn import DeformableConv2d
from .ltae import LTAE2d
from .utae import Temporal_Aggregator
from .ref_arch_util import LayerNorm, to_3d, to_4d, Mlp, Mutual_Attention

class BaseUNet(nn.Module):
    def __init__(self, base_num_channels, num_encoders, num_residual_blocks,
                 num_output_channels, skip_type, norm, use_upsample_conv,
                 num_bins, recurrent_block_type=None, kernel_size=5,
                 channel_multiplier=2):
        super(BaseUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type

        self.encoder_input_sizes = [int(self.base_num_channels * pow(channel_multiplier, i)) for i in range(self.num_encoders)]
        self.encoder_output_sizes = [int(self.base_num_channels * pow(channel_multiplier, i + 1)) for i in range(self.num_encoders)]
        self.max_num_channels = self.encoder_output_sizes[-1]
        self.skip_ftn = eval('skip_' + skip_type)
        print('Using skip: {}'.format(self.skip_ftn))
        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer
        assert(self.num_output_channels > 0)
        print(f'Kernel size {self.kernel_size}')
        print(f'Skip type {self.skip_type}')
        print(f'norm {self.norm}')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(self.UpsampleLayer(
                input_size if self.skip_type == 'sum' else 2 * input_size,
                output_size, kernel_size=self.kernel_size,
                padding=self.kernel_size // 2, norm=self.norm))
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                         num_output_channels, 1, activation=None, norm=norm)


class WNet(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    One decoder for flow and one for image.
    """

    def __init__(self, unet_kwargs):
        unet_kwargs['num_output_channels'] = 3
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.image_decoders = self.build_decoders()
        self.flow_decoders = self.build_decoders()
        self.image_pred = self.build_prediction_layer(num_output_channels=1)
        self.flow_pred = self.build_prediction_layer(num_output_channels=2)
        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        flow_activations = x
        for i, decoder in enumerate(self.flow_decoders):
            flow_activations = decoder(self.skip_ftn(flow_activations, blocks[self.num_encoders - i - 1]))
        image_activations = x
        for i, decoder in enumerate(self.image_decoders):
            image_activations = decoder(self.skip_ftn(image_activations, blocks[self.num_encoders - i - 1]))

        # tail
        flow = self.flow_pred(self.skip_ftn(flow_activations, head))
        image = self.image_pred(self.skip_ftn(image_activations, head))

        output_dict = {'image': image, 'flow': flow}

        return output_dict


class UNetFlow(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        unet_kwargs['num_output_channels'] = 3
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(num_output_channels=3)
        self.states = [None] * self.num_encoders
        self.merge_c = []
        self.merge_c.append(self.base_num_channels)
        self.merge_c += self.encoder_output_sizes
        self.merge_c.pop()

    # hard coded states to use nn.dataparallel
    def forward(self, x, state1, state2, state3):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            if i == 0:
                t_state = state1
            elif i == 1:
                t_state = state2
            elif i == 2:
                t_state = state3 
            x, state = encoder(x, t_state)
            if i == 0:
                t_state_out1 = state
            elif i == 1:
                t_state_out2 = state
            elif i == 2:
                t_state_out3 = state
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img_flow = self.pred(self.skip_ftn(x, head))

        blocks.insert(0, head)
        blocks.pop()
        output_dict = {'image': img_flow[:, 0:1, :, :], 'flow': img_flow[:, 1:3, :, :], 'evt_feat': blocks, 'state1': t_state_out1, 'state2': t_state_out2, 'state3': t_state_out3}

        return output_dict


class UNetFlowNoRecur(BaseUNet):
    """
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        unet_kwargs['num_output_channels'] = 3
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(ConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(num_output_channels=3)

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img_flow = self.pred(self.skip_ftn(x, head))

        output_dict = {'image': img_flow[:, 0:1, :, :], 'flow': img_flow[:, 1:3, :, :]}

        return output_dict


class UNetRecurrent(BaseUNet):
    """
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """
    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        print(f'Using {self.final_activation} final activation')
        unet_kwargs['num_output_channels'] = 1
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)
        return {'image': img}


class UNet(BaseUNet):
    """
    UNet architecture. Symmetric, skip connections on every encoding layer.
    """
    def __init__(self, unet_kwargs):
        super().__init__(**unet_kwargs)
        self.encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins  # since there is no self.head!
            self.encoders.append(ConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = ConvLayer(self.base_num_channels, self.num_output_channels, kernel_size=1, activation=None)

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        return self.pred(x)


class UNetFlow_encoder_backward(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: a list of multi-scale features, each N x C x H_f x W_f
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        output_dict = {'evt_feat': blocks, 'states': self.states}

        return output_dict


class UNetFlow_encoder_forward(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    the forward encoder takes backward features at each recurrent iter
    """

    def __init__(self, unet_kwargs):
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        self.forward_backward_merge = nn.ModuleList()
  
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))
            if self.skip_type == 'sum':
                self.forward_backward_merge.append(
                                  ConvLayer(output_size, output_size,
                                  kernel_size=3, stride=1,
                                  padding=1)  # N x 2*C x H x W -> N x C x H x W
                                  )
            else:
                self.forward_backward_merge.append(
                                  ConvLayer(2*output_size, output_size,
                                  kernel_size=3, stride=1,
                                  padding=1)  # N x 2*C x H x W -> N x C x H x W
                                  )



        self.states = [None] * self.num_encoders
        # used to merge with backward feat
        self.skip_ftn = eval('skip_' + self.skip_type)

    def forward(self, x, backward_feat):
        """
        :param x: N x num_input_channels x H x W
        :param backward_feat: a list of multi-scale features of backward at current time
        :return: a list of multi-scale features, each N x C x H_f x W_f
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            x = self.forward_backward_merge[i](self.skip_ftn(x, backward_feat[i]))
            blocks.append(x)
            self.states[i] = state

        output_dict = {'evt_feat': blocks, 'states': self.states}

        return output_dict


class UNetFlowNoRecur_vis(BaseUNet):
    """
    Symmetric, skip connections on every encoding layer.
    receive deblur event feature, also evt reconstruction feature
    """

    def __init__(self, unet_kwargs):
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        self.evt_feat_merge = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(UNetConvBlock(
                input_size, output_size, True, 0.2))
            if self.skip_type == 'sum':
                self.evt_feat_merge.append(
                                  UNetConvBlock(output_size, output_size,
                                  False,0.2)  # N x C x H x W -> N x C x H x W
                                  )
            else:
                # 3 * output_size since we have three data sources
                # vis image, deblur evt (within exposure window), rec evt
                self.evt_feat_merge.append(
                                  UNetConvBlock(3*output_size, output_size,
                                  False,0.2)  # N x 3*C x H x W -> N x C x H x W
                                  )

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(num_output_channels=1)


    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(UNetUpBlock(
                input_size if self.skip_type == 'sum' else 2 * input_size,
                output_size, 0.2))
        return decoders


    def forward(self, x, evt_rec_feats, evt_deblur_feats):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            merge_feat = self.skip_ftn(x, evt_rec_feats[i])
            merge_feat = self.skip_ftn(merge_feat, evt_deblur_feats[i])
            x = self.evt_feat_merge[i](merge_feat)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img_flow = self.pred(self.skip_ftn(x, head))

        output_dict = {'image': img_flow[:, 0:1, :, :]}

        return output_dict


class CorrFeatSplittingModules(nn.Module):
    # directly take evt reconstruction and efnet and get channel information from them
    def __init__(self, model_evt_rec, efnet):
        super(CorrFeatSplittingModules, self).__init__()
        # take min, although generaly both are 3
        self.num_stages = min(model_evt_rec.num_encoders, efnet.depth)
        # number of feature channels
        self.evt_rec_num_channels = model_evt_rec.unetflow.encoder_output_sizes
        self.efnet_num_channels = [(2**i)*efnet.wf for i in range(efnet.depth)]
        self.splitting_modules = [SplittingModule(self.evt_rec_num_channels[i], self.efnet_num_channels[i]) for i in range(self.num_stages)]
        self.splitting_modules = nn.ModuleList(self.splitting_modules)

    def forward(self, f_evt_rec, f_efnet, stage_idx):
        common_feat, diff_feat = self.splitting_modules[stage_idx](f_evt_rec, f_efnet)
        return common_feat, diff_feat

# a temp implementation
class SplittingModule(nn.Module):
    # directly take evt reconstruction and efnet and get channel information from them
    def __init__(self, evt_rec_feat_channels, efnet_feat_channels):
        super(SplittingModule, self).__init__()

        self.merge_conv = UNetConvBlock(evt_rec_feat_channels+efnet_feat_channels, efnet_feat_channels,False,0.2) 
        self.split_conv = UNetConvBlock(evt_rec_feat_channels+efnet_feat_channels, efnet_feat_channels,False,0.2) 

    def forward(self, f_evt_rec, f_efnet):
        # interpolate spatially
        if f_evt_rec.shape[2] != f_efnet.shape[2] or f_evt_rec.shape[3] != f_efnet.shape[3]:
            f_evt_rec = f.interpolate(f_evt_rec, [f_efnet.shape[2], f_efnet.shape[3]], mode='bilinear')
        cat_feat = torch.cat((f_evt_rec,f_efnet),1)
        # keep it residual
        common_feat = f_efnet + self.merge_conv(cat_feat)
        diff_feat = self.merge_conv(cat_feat)
        return common_feat, diff_feat


class CorrFeatSplittingModules(nn.Module):
    # directly take evt reconstruction and efnet and get channel information from them
    def __init__(self, model_evt_rec, efnet):
        super(CorrFeatSplittingModules, self).__init__()
        # take min, although generaly both are 3
        self.num_stages = min(model_evt_rec.num_encoders, efnet.depth)
        # number of feature channels
        self.evt_rec_num_channels = model_evt_rec.unetflow.encoder_output_sizes
        self.efnet_num_channels = [(2**i)*efnet.wf for i in range(efnet.depth)]
        self.splitting_modules = [SplittingModule(self.evt_rec_num_channels[i], self.efnet_num_channels[i]) for i in range(self.num_stages)]
        self.splitting_modules = nn.ModuleList(self.splitting_modules)

    def forward(self, f_evt_rec, f_efnet, stage_idx):
        common_feat, diff_feat = self.splitting_modules[stage_idx](f_evt_rec, f_efnet)
        return common_feat, diff_feat

# a temp implementation
class SplittingModule(nn.Module):
    # directly take evt reconstruction and efnet and get channel information from them
    def __init__(self, evt_rec_feat_channels, efnet_feat_channels):
        super(SplittingModule, self).__init__()

        self.merge_conv = UNetConvBlock(evt_rec_feat_channels+efnet_feat_channels, efnet_feat_channels,False,0.2) 
        self.split_conv = UNetConvBlock(evt_rec_feat_channels+efnet_feat_channels, efnet_feat_channels,False,0.2) 

    def forward(self, f_evt_rec, f_efnet):
        # interpolate spatially
        if f_evt_rec.shape[2] != f_efnet.shape[2] or f_evt_rec.shape[3] != f_efnet.shape[3]:
            f_evt_rec = f.interpolate(f_evt_rec, [f_efnet.shape[2], f_efnet.shape[3]], mode='bilinear')
        cat_feat = torch.cat((f_evt_rec,f_efnet),1)
        # keep it residual
        common_feat = f_efnet + self.merge_conv(cat_feat)
        diff_feat = self.merge_conv(cat_feat)
        return common_feat, diff_feat


class DConvModules(nn.Module):
    def __init__(self, model_evt_rec, efnet):
        super(DConvModules, self).__init__()
        # take min, although generaly both are 3
        self.num_stages = min(model_evt_rec.num_encoders, efnet.depth)
        # number of feature channels
        self.evt_rec_num_channels = model_evt_rec.unetflow.merge_c
        self.efnet_num_channels = [(2**i)*efnet.wf for i in range(efnet.depth)]
        self.dconv_modules = [DeformableConv2d(self.evt_rec_num_channels[i]+self.efnet_num_channels[i], self.efnet_num_channels[i], offset_channels=self.evt_rec_num_channels[i]+self.efnet_num_channels[i]) for i in range(self.num_stages)]
        self.dconv_modules = nn.ModuleList(self.dconv_modules)
        self.activation = nn.LeakyReLU(0.2)
        LayerNorm_type = 'WithBias'
        self.norm_evt_rec = [LayerNorm(self.evt_rec_num_channels[i], LayerNorm_type) for i in range(self.num_stages)]
        self.norm_evt_rec = nn.ModuleList(self.norm_evt_rec)
        self.norm_efnet = [LayerNorm(self.efnet_num_channels[i], LayerNorm_type) for i in range(self.num_stages)]
        self.norm_efnet = nn.ModuleList(self.norm_efnet)
        self.norm_aligned_feat = [LayerNorm(self.efnet_num_channels[i], LayerNorm_type) for i in range(self.num_stages)]
        self.norm_aligned_feat = nn.ModuleList(self.norm_aligned_feat)
        self.ffn = [Mlp(in_features=self.efnet_num_channels[i], hidden_features=2*self.efnet_num_channels[i], act_layer=nn.GELU, drop=0.) for i in range(self.num_stages)]
        self.ffn = nn.ModuleList(self.ffn)
        self.norm2 = [nn.LayerNorm(self.efnet_num_channels[i]) for i in range(self.num_stages)]
        self.norm2 = nn.ModuleList(self.norm2)

    def forward(self, f_evt_rec, f_efnet, stage_idx):
        f_efnet_ori = f_efnet
        f_evt_rec = self.norm_evt_rec[stage_idx](f_evt_rec)
        f_efnet = self.norm_efnet[stage_idx](f_efnet)
        offset_x = torch.cat((f_evt_rec,f_efnet), 1)
        aligned_feat = self.dconv_modules[stage_idx](offset_x)
        aligned_feat = self.activation(aligned_feat)
        aligned_feat = self.norm_aligned_feat[stage_idx](aligned_feat)
        fused = aligned_feat + f_efnet_ori
        b, c , h, w = fused.shape
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn[stage_idx](self.norm2[stage_idx](fused))
        fused = to_4d(fused, h, w)

        return aligned_feat


class TempCrossAttnModules(nn.Module):
    def __init__(self, model_evt_rec, efnet):
        super(TempCrossAttnModules, self).__init__()
        # take min, although generaly both are 3
        self.num_stages = min(model_evt_rec.num_encoders, efnet.depth)
        # number of feature channels
        self.evt_rec_num_channels = model_evt_rec.unetflow.encoder_output_sizes
        self.efnet_num_channels = [(2**i)*efnet.wf for i in range(efnet.depth)]
        
        # we have only one attention module
        # that just works on the last stage
        self.tcrossattn_module = []
        for i in range(self.num_stages):
            in_channels = self.efnet_num_channels[i]
            n_head = 4
            d_k = 32
            return_att = True
            d_in = self.efnet_num_channels[i]
            self.tcrossattn_module.append(LTAE2d(in_channels=in_channels,n_head=n_head, d_k=d_k, mlp=[d_in,d_in], return_att=return_att))
        self.tcrossattn_module = nn.ModuleList(self.tcrossattn_module)
        self.temp_aggreg = Temporal_Aggregator(mode="att_group")
        self.align_1x1_conv = [ConvLayer(self.efnet_num_channels[i], self.efnet_num_channels[i], 1, activation=None, norm=None) for i in range(self.num_stages)]
        self.align_1x1_conv = nn.ModuleList(self.align_1x1_conv)
        self.register_parameter("mu", nn.parameter.Parameter(0.0*torch.ones(self.num_stages,)))

    # do it al at once
    # first compute cross attn for the last stage
    # then use the same (upsampled) attn weights to 
    # modulate first 2 stages's feat
    # finally use 1x1 conv to merge
    # return a list of tensor similar to f_efnet_list
    def forward(self, f_evt_rec_stage_T_feats, f_efnet_list):

        merge_feats = []

        for i in range(self.num_stages):
            # N, T, C, H, W
            last_f_evt_rec = f_evt_rec_stage_T_feats[i]
            # N, C, H, W
            last_f_efnet = f_efnet_list[i]
            out, attn = self.tcrossattn_module[i](last_f_evt_rec, last_f_efnet)
            f = out
            merge_f = self.align_1x1_conv[i](f)
            merge_f = (1-self.mu[i]) * f_efnet_list[i] + self.mu[i] * merge_f
            merge_feats.append(merge_f)
        #out: N,C,H,W
        #attn: n_head, N, T, H, W
        #out, attn = self.tcrossattn_module(last_f_evt_rec, last_f_efnet)
        #out_feats = []
        #for i in range(self.num_stages-1):
        #    f = self.temp_aggreg(f_evt_rec_stage_T_feats[i], attn_mask=attn)
        #    out_feats.append(f)
        #out_feats.append(out)

        #merge
        #merge_feats = f_efnet_list[0:-1]
        # for i in range(self.num_stages):
        #     f = out_feats[i]
        #     merge_f = self.align_1x1_conv[i](f)
        #     merge_f = (1-self.mu[i]) * f_efnet_list[i] + self.mu[i] * merge_f
        #     merge_feats.append(merge_f)

        #     ## normalize both f and f_efnet_list[i]
        #     #merge_f = torch.cat((f,f_efnet_list[i]), 1)
        #     #merge_f = (1-self.mu[i]) * f_efnet_list[i] + self.mu[i] * merge_f
        #     #merge_feats.append(merge_f)

        return merge_feats


class LSTMEncoder(nn.Module):
    def __init__(self, model_evt_rec, efnet):
        super(LSTMEncoder, self).__init__()
        # take min, although generaly both are 3
        self.num_stages = min(model_evt_rec.num_encoders, efnet.depth)
        # number of feature channels
        self.evt_rec_num_channels = model_evt_rec.unetflow.merge_c
        self.efnet_num_channels = [(2**i)*efnet.wf for i in range(efnet.depth)]
        self.encoders = nn.ModuleList()
        for i in range(self.num_stages):
            input_size = self.evt_rec_num_channels[i]
            # //2 since we have a forward and a backward
            output_size = self.efnet_num_channels[i] // 2
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=3, stride=1,
                padding=1,
                recurrent_block_type="convlstm", norm=None))

    def forward(self, f_evt_rec, stage_idx, reverse=False):
        T = f_evt_rec.shape[1]
        state = None
        if not reverse:
            for i in range(T):
                f_evt_rec_ti = f_evt_rec[:,i,:,:,:]
                rnn_f, state = self.encoders[stage_idx](f_evt_rec_ti, state)
        else:
            for i in range(T-1,-1,-1):
                f_evt_rec_ti = f_evt_rec[:,i,:,:,:]
                rnn_f, state = self.encoders[stage_idx](f_evt_rec_ti, state)
        return rnn_f


## cross-task channel attention (CTCA)
class CTCA(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(CTCA, self).__init__()
        self.attn = Mutual_Attention(dim, num_heads, bias)
        self.norm_task1 = LayerNorm(dim, LayerNorm_type)
        self.norm_task2 = LayerNorm(dim, LayerNorm_type)
        self.norm = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, feat_task1, feat_task2):
        b, c , h, w = feat_task1.shape
        feat = feat_task1 + self.attn(self.norm_task1(feat_task1), self.norm_task2(feat_task2)) 
        feat = to_3d(feat) 
        feat = feat + self.ffn(self.norm(feat))
        feat = to_4d(feat, h, w)

        return feat

class CTCA_stages(nn.Module):
    def __init__(self, model_evt_rec, efnet):
        super(CTCA_stages, self).__init__()
        # take min, although generaly both are 3
        self.num_stages = min(model_evt_rec.num_encoders, efnet.depth)
        # number of feature channels
        self.evt_rec_num_channels = model_evt_rec.unetflow.merge_c
        self.efnet_num_channels = [(2**i)*efnet.wf for i in range(efnet.depth)]
        self.num_heads = [1,2,4]
        self.encoders = nn.ModuleList()
        for i in range(self.num_stages):
            output_size = self.efnet_num_channels[i]
            self.encoders.append(CTCA(output_size, self.num_heads[i]))

    def forward(self, f_evt_rec, f_efnet, stage_idx):
        return self.encoders[stage_idx](f_efnet, f_evt_rec)
