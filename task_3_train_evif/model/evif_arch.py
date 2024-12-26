import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
from .network_swinir import PatchEmbed, PatchUnEmbed, BasicLayer
from info_nce import InfoNCE


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,padding_mode="reflect"),
        )
    def forward(self, x):
        out = self.conv(x)
        return out+x


class ConvEncoder(nn.Module):
    def __init__(self, input_nc=1, channels=[8, 16, 32]):
        super(ConvEncoder, self).__init__()
        self.first_conv = ResBlock(input_nc, channels[0])
        self.down_convs = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        self.relu = nn.ReLU()
        self.channels = channels
        
        for i in range(len(channels) - 1):
            self.down_convs.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, bias=False, padding_mode="reflect"))
            self.res_blocks.append(ResBlock(channels[i + 1], channels[i + 1]))
        
        self.num_blocks = len(channels) - 1

    def forward(self, x):
        x = self.first_conv(x)
        for i in range(self.num_blocks):
            x = self.down_convs[i](x)
            x = self.relu(x)
            x = self.res_blocks[i](x)
        return x
    
    def get_resblock_features(self, x):
        x = self.first_conv(x)
        resblock_features = [x]
        for i in range(self.num_blocks):
            x = self.down_convs[i](x)
            x = self.relu(x)
            x = self.res_blocks[i](x)
            resblock_features.append(x)
        return resblock_features


class ConvDecoder(nn.Module):
    def __init__(self, channels=[32, 16, 8], output_nc=1):
        super(ConvDecoder, self).__init__()
        self.up_convs = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        self.reduce_convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.channels = channels
        
        for i in range(len(channels) - 1):
            self.up_convs.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1, bias=False)
            ))
            self.res_blocks.append(ResBlock(channels[i + 1], channels[i + 1]))
            self.reduce_convs.append(DepthwiseSeparableConv(2 * channels[i + 1], channels[i + 1]))
        
        self.final_conv = nn.Conv2d(channels[-1], output_nc, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect")
        self.num_blocks = len(channels) - 1

    def forward(self, encoder_features):
        x = encoder_features[-1]
        for i in range(self.num_blocks):
            x = self.up_convs[i](x)
            x = self.relu(x)
            x = torch.cat((x, encoder_features[self.num_blocks - 1 - i]), dim=1)
            x = self.reduce_convs[i](x)
            x = self.res_blocks[i](x)
        x = self.final_conv(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size//2, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x

class FusionChannelAttnConv(nn.Module):
    def __init__(self, dim1, dim2, ratio=2):
        super().__init__()
        dim = dim1 + dim2
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // ratio, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(dim // ratio, dim // ratio, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(dim // ratio, dim, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(dim // ratio, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        B, C, H, W = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)

class FusionTransformers(nn.Module):
    def __init__(self, feat_nc=64, img_size=256, patch_size=4, embed_dim=96, window_size=4):
        super().__init__()
        norm_layer=nn.LayerNorm
        self.patch_norm = True
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=feat_nc, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)   
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)   
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.norm = nn.LayerNorm(embed_dim)
        self.window_size = window_size
        # information filter
        self.basicLayer1 = BasicLayer(dim=feat_nc,
                                     input_resolution=(patches_resolution[0], patches_resolution[1]),
                                     depth=2,
                                     num_heads=2,
                                     window_size=window_size,
                                     mlp_ratio=2.,
                                     qkv_bias=True, qk_scale=None,
                                     drop=0., attn_drop=0.,
                                     drop_path=0.,
                                     norm_layer=norm_layer,
                                     downsample=None,
                                     use_checkpoint=False)
        # information filter
        self.basicLayer2 = BasicLayer(dim=feat_nc,
                                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                                    depth=2,
                                    num_heads=2,
                                    window_size=window_size,
                                    mlp_ratio=2.,
                                    qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0.,
                                    drop_path=0.,
                                    norm_layer=norm_layer,
                                    downsample=None,
                                    use_checkpoint=False)
        # fusion transformer
        self.basicLayer3 = BasicLayer(dim=feat_nc,
                                     input_resolution=(patches_resolution[0], patches_resolution[1]),
                                     depth=2,
                                     num_heads=2,
                                     window_size=window_size,
                                     mlp_ratio=2.,
                                     qkv_bias=True, qk_scale=None,
                                     drop=0., attn_drop=0.,
                                     drop_path=0.,
                                     norm_layer=norm_layer,
                                     downsample=None,
                                     use_checkpoint=False)
        
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    # information filter
    def forward(self, f_inf, f_vis):
        H, W = f_inf.shape[2:]
        f_inf = self.check_image_size(f_inf)
        f_vis = self.check_image_size(f_vis)
        encode_size_inf = (f_inf.shape[2],f_inf.shape[3])
        f_inf_emb = self.patch_embed(f_inf)
        f_inf_filtered = self.basicLayer1(f_inf_emb, encode_size_inf)
        f_inf_filtered = self.norm(f_inf_filtered)
        f_inf_filtered = self.patch_unembed(f_inf_filtered, encode_size_inf)
        #no residual connection experiments/evif_mi_attn_pointwise4_no_residual/models/evif/1101_134303
        #f_inf_filtered = f_inf_filtered + f_inf 

        encode_size_vis = (f_vis.shape[2],f_vis.shape[3])
        f_vis_emb = self.patch_embed(f_vis)
        f_vis_filtered = self.basicLayer2(f_vis_emb, encode_size_vis)
        f_vis_filtered = self.norm(f_vis_filtered)
        f_vis_filtered = self.patch_unembed(f_vis_filtered, encode_size_vis)
        #f_vis_filtered = f_vis_filtered + f_vis

        return f_inf_filtered[:, :, :H, :W], f_vis_filtered[:, :, :H, :W]
    
    # fuse
    def fuse(self, f_inf, f_vis):
        H, W = f_inf.shape[2:]
        f_inf = self.check_image_size(f_inf)
        f_vis = self.check_image_size(f_vis)
        # cat
        f = f_inf + f_vis
        # fuse
        encode_size_f = (f.shape[2],f.shape[3])
        f_emb = self.patch_embed(f)
        f_fused = self.basicLayer3(f_emb, encode_size_f)
        f_fused = self.norm(f_fused)
        f_fused = self.patch_unembed(f_fused, encode_size_f)
        return f_fused[:, :, :H, :W]
    
    
class MI_Minimization_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size, input_size=256):
        super(MI_Minimization_reg, self).__init__()
        self.input_channels = input_channels
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, input_channels, kernel_size=3, stride=1, padding=1),
        )
        self.channel = channels

        self.adaptive_pool = nn.AdaptiveAvgPool2d((input_size, input_size))

        self.fc1 = nn.Linear(input_channels*input_size*input_size, latent_size)
        self.fc2 = nn.Linear(input_channels*input_size*input_size, latent_size)

        self.attn_conv_vis = nn.Conv2d(2, 1, kernel_size=3, padding=3 // 2, bias=False)
        self.attn_conv_inf = nn.Conv2d(2, 1, kernel_size=3, padding=3 // 2, bias=False)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.CE = torch.nn.BCELoss(reduction='sum')

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, inf_feat, vis_feat):
        # import ipdb ; ipdb.set_trace()
        # import matplotlib
        # from pytorchvis.visualize_layers import VisualizeLayers
        # vis = VisualizeLayers(self,layers='conv')
        # for i in range(inf_feat.shape[0]):
        #     vis.plot_featuremaps(inf_feat[i:i+1,:,:,:].detach().cpu(),name='fmaps_inf_in_%d'%i,savefig=True)
        #     vis.plot_featuremaps(vis_feat[i:i+1,:,:,:].detach().cpu(),name='fmaps_vis_in_%d'%i,savefig=True)

        inf_feat = self.layers(inf_feat)
        vis_feat = self.layers(vis_feat)

        inf_avg_out = torch.mean(inf_feat, dim=1, keepdim=True)
        vis_avg_out = torch.mean(vis_feat, dim=1, keepdim=True)
        inf_max_out, _ = torch.max(inf_feat, dim=1, keepdim=True)
        vis_max_out, _ = torch.max(vis_feat, dim=1, keepdim=True)
        inf_feat_am = torch.cat([inf_avg_out, inf_max_out], dim=1)
        vis_feat_am = torch.cat([vis_avg_out, vis_max_out], dim=1)
        inf_feat_attn = self.attn_conv_inf(inf_feat_am)
        vis_feat_attn = self.attn_conv_vis(vis_feat_am)
        inf_feat_attn = self.sigmoid(inf_feat_attn)
        vis_feat_attn = self.sigmoid(vis_feat_attn)

        # for i in range(inf_feat.shape[0]):
        #     vis.plot_featuremaps(inf_feat[i:i+1,:,:,:].detach().cpu(),name='fmaps_inf_out_%d'%i,savefig=True)
        #     vis.plot_featuremaps(vis_feat[i:i+1,:,:,:].detach().cpu(),name='fmaps_vis_out_%d'%i,savefig=True)
            
        # for i in range(inf_feat_attn.shape[0]):
        #     vis.plot_featuremaps(inf_feat_attn[i:i+1,:,:,:].detach().cpu(),name='fmaps_inf_att_%d'%i,savefig=True)
        #     vis.plot_featuremaps(vis_feat_attn[i:i+1,:,:,:].detach().cpu(),name='fmaps_vis_att_%d'%i,savefig=True)
        
        vis_feat = self.adaptive_pool(vis_feat)
        inf_feat = self.adaptive_pool(inf_feat)
        vis_feat = vis_feat.contiguous().view(vis_feat.shape[0], -1)
        inf_feat = inf_feat.contiguous().view(inf_feat.shape[0], -1)
        mu_vis = self.fc1(vis_feat)
        logvar_vis = self.fc2(vis_feat)
        mu_inf = self.fc1(inf_feat)
        logvar_inf = self.fc2(inf_feat)

        mu_inf = self.tanh(mu_inf)
        mu_vis = self.tanh(mu_vis)
        logvar_inf = self.tanh(logvar_inf)
        logvar_vis = self.tanh(logvar_vis)
        mu_inf = mu_inf.contiguous().view(mu_inf.shape[0], -1)
        mu_vis = mu_vis.contiguous().view(mu_vis.shape[0], -1)
        logvar_inf = logvar_inf.contiguous().view(logvar_inf.shape[0], -1)
        logvar_vis = logvar_vis.contiguous().view(logvar_vis.shape[0], -1)
        z_vis = self.reparametrize(mu_vis, logvar_vis)
        dist_vis = Independent(Normal(loc=mu_vis, scale=torch.exp(logvar_vis)), 1)
        z_inf = self.reparametrize(mu_inf, logvar_inf)
        dist_inf = Independent(Normal(loc=mu_inf, scale=torch.exp(logvar_inf)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_vis, dist_inf)) + torch.mean(
            self.kl_divergence(dist_inf, dist_vis))
        z_vis_norm = torch.sigmoid(z_vis)
        z_inf_norm = torch.sigmoid(z_inf)
        ce_vis_inf = self.CE(z_vis_norm, z_inf_norm.detach())
        ce_inf_vis = self.CE(z_inf_norm, z_vis_norm.detach())
        latent_loss = ce_vis_inf + ce_inf_vis - bi_di_kld

        output_dict = {}
        output_dict['latent_loss'] = latent_loss
        output_dict['inf_feat_attn'] = inf_feat_attn
        output_dict['vis_feat_attn'] = vis_feat_attn

        return output_dict

class SobelConv(nn.Module):
    def __init__(self, input_channels):
        super(SobelConv, self).__init__()
        self.input_channels = input_channels
        self.sobel_kernel = self.create_sobel_kernel(input_channels)
        
        # Create a fixed convolutional layer
        self.sobel_layer = nn.Conv2d(input_channels, 2 * input_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_layer.weight = nn.Parameter(self.sobel_kernel, requires_grad=False)

    def create_sobel_kernel(self, input_channels):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        # Initialize weight tensor
        kernel = torch.zeros((2 * input_channels, input_channels, 3, 3), dtype=torch.float32)

        for i in range(input_channels):
            kernel[2 * i, i, :, :] = torch.from_numpy(Kx)  # Gradient in x
            kernel[2 * i + 1, i, :, :] = torch.from_numpy(Ky)  # Gradient in y
        
        return kernel

    def forward(self, x):
        return self.sobel_layer(x)


class Residual_Block_mi_max(nn.Module):
        def __init__(self, ch_in, ch_out, identity=None):
                super(Residual_Block_mi_max, self).__init__()

                self.in_channels = ch_in
                self.out_channels = ch_out
                self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                                                           out_channels=self.in_channels,
                                                           kernel_size=1,
                                                           stride=1,
                                                           padding=0,
                                                           bias=False)
                self.bn1 = nn.BatchNorm2d(self.in_channels)


                self.conv2 = nn.Conv2d(in_channels=self.in_channels,
                                                           out_channels=self.out_channels,
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1,
                                                           bias=False)
                self.bn2 = nn.BatchNorm2d(self.out_channels)


                self.identity_block = nn.Conv2d(in_channels=self.in_channels,
                                                out_channels=self.out_channels ,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=False)
                self.identity = identity
                self.lrelu = nn.LeakyReLU(inplace=True)

        def forward(self, x):
                residual = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.lrelu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                if self.identity:
                        residual = self.identity_block(x)
                out += residual
                out = self.lrelu(out)
                return out


class mi_max_encoder(nn.Module):
        def __init__(self, in_channels, out_channels):
                super(mi_max_encoder, self).__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels

                self.res_block0 = Residual_Block_mi_max(self.in_channels, self.out_channels, identity=True)
                self.res_block1 = Residual_Block_mi_max(self.out_channels, 2*self.out_channels, identity=True)
                self.res_block2 = Residual_Block_mi_max(2*self.out_channels, self.out_channels, identity=True)
                self.res_block3 = Residual_Block_mi_max(self.out_channels, self.out_channels, identity=True)


        def forward(self, x):
                feat = self.res_block0(x)
                feat = self.res_block1(feat)
                feat = self.res_block2(feat)
                feat = self.res_block3(feat)
                return feat


class MI_Maximization_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size, input_size=256, sobel=False):
        super(MI_Maximization_reg, self).__init__()
        self.input_channels = input_channels
        if sobel:
            self.layers = nn.Sequential(
                SobelConv(input_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(2*input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layers = nn.Identity()

        self.channel = channels
        self.adaptive_pool = nn.AdaptiveAvgPool2d((input_size, input_size))
        self.fc = nn.Sequential(*[nn.Linear(input_channels, input_channels), nn.BatchNorm1d(input_channels), nn.LeakyReLU(inplace=True), nn.Linear(input_channels, latent_size)])
        self.NCE = InfoNCE(temperature=1.0, reduction='none')

    def forward(self, inp_feat, fus_feat):
        # import ipdb ; ipdb.set_trace()
        # import matplotlib
        # from pytorchvis.visualize_layers import VisualizeLayers
        # vis = VisualizeLayers(self,layers='conv')
        # for i in range(inf_feat.shape[0]):
        #     vis.plot_featuremaps(inf_feat[i:i+1,:,:,:].detach().cpu(),name='fmaps_inf_in_%d'%i,savefig=True)
        #     vis.plot_featuremaps(vis_feat[i:i+1,:,:,:].detach().cpu(),name='fmaps_vis_in_%d'%i,savefig=True)

        inp_feat = self.layers(inp_feat)
        fus_feat = self.layers(fus_feat)

        # for i in range(inf_feat.shape[0]):
        #     vis.plot_featuremaps(inf_feat[i:i+1,:,:,:].detach().cpu(),name='fmaps_inf_out_%d'%i,savefig=True)
        #     vis.plot_featuremaps(vis_feat[i:i+1,:,:,:].detach().cpu(),name='fmaps_vis_out_%d'%i,savefig=True)
            
        # for i in range(inf_feat_attn.shape[0]):
        #     vis.plot_featuremaps(inf_feat_attn[i:i+1,:,:,:].detach().cpu(),name='fmaps_inf_att_%d'%i,savefig=True)
        #     vis.plot_featuremaps(vis_feat_attn[i:i+1,:,:,:].detach().cpu(),name='fmaps_vis_att_%d'%i,savefig=True)
        
        inp_feat = self.adaptive_pool(inp_feat) # N, dim, inp_size, inp_size
        fus_feat = self.adaptive_pool(fus_feat)

        N = inp_feat.shape[0]
        dim = inp_feat.shape[1]
        inp_size1 = inp_feat.shape[2]
        inp_size2 = inp_feat.shape[3]

        inp_feat = torch.permute(inp_feat, (0, 2, 3, 1)) # N, inp_size, inp_size, dim
        fus_feat = torch.permute(fus_feat, (0, 2, 3, 1)) # N, inp_size, inp_size, dim
        inp_feat = inp_feat.contiguous().view(-1, dim) # N x inp_size x inp_size, dim
        fus_feat = fus_feat.contiguous().view(-1, dim) # N x inp_size x inp_size, dim

        try:
            inp_feat_fc = self.fc(inp_feat)
            fus_feat_fc = self.fc(fus_feat)
            dim = inp_feat_fc.shape[1]
            inp_feat_fc = inp_feat_fc.contiguous().view(N, -1, dim) # N,  inp_size x inp_size, dim
            fus_feat_fc = fus_feat_fc.contiguous().view(N, -1, dim) # N,  inp_size x inp_size, dim
            batchSize = inp_feat_fc.shape[1] # B = num feat pixels
            num_samples = 200
            weight = torch.ones(batchSize).to(inp_feat_fc.device)
            index = torch.multinomial(weight, num_samples) # index: num_samples index
            expanded_index = index.unsqueeze(0).unsqueeze(-1).expand(N, -1, dim) # N, num_samples, dim
            selected_anchor_samples = torch.gather(fus_feat_fc, 1, expanded_index) # Shape: (N, num_samples, dim)
            selected_pos_samples = torch.gather(inp_feat_fc, 1, expanded_index) # Shape: (N, num_samples, dim)
            inp_feat_fc = selected_pos_samples.contiguous().view(-1, dim) # N x num_samples, dim
            fus_feat_fc = selected_anchor_samples.contiguous().view(-1, dim) # N x num_samples, dim
            inp_feat_fc_norm = F.normalize(inp_feat_fc, dim=1)
            fus_feat_fc_norm = F.normalize(fus_feat_fc, dim=1)
            dist = torch.bmm(fus_feat_fc_norm.view(fus_feat_fc.shape[0], 1, -1), inp_feat_fc_norm.view(inp_feat_fc.shape[0], -1, 1))
            dist = dist.view(fus_feat_fc.shape[0], 1)
            latent_loss = self.NCE(fus_feat_fc, inp_feat_fc)  # shape: torch.Size([B])
        except Exception as e:
            print(e)
            dist = torch.zeros((inp_feat.shape[0], 1), requires_grad=True).to(inp_feat.device)
            latent_loss = torch.zeros((inp_feat.shape[0]), requires_grad=True).to(inp_feat.device)

        output_dict = {}
        output_dict['latent_loss'] = latent_loss
        output_dict['dist'] = dist 

        return output_dict


