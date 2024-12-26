import torch
import torch.nn.functional as F
# local modules
from PerceptualSimilarity import models
from utils import loss
import numpy as np
from .pytorch_msssim import msssim
from info_nce import InfoNCE
from .ref_loss_SeAFusion import Fusionloss



class combined_perceptual_loss():
    def __init__(self, weight=1.0, use_gpu=True):
        """
        Flow wrapper for perceptual_loss
        """
        self.loss = perceptual_loss(weight=1.0, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred_img, pred_flow, target_img, target_flow):
        """
        image is tensor of N x 2 x H x W, flow of N x 2 x H x W
        These are concatenated, as perceptualLoss expects N x 3 x H x W.
        """
        pred = torch.cat([pred_img, pred_flow], dim=1)
        target = torch.cat([target_img, target_flow], dim=1)
        dist = self.loss(pred, target, normalize=False)
        return dist * self.weight


class warping_flow_loss():
    def __init__(self, weight=1.0, L0=1):
        assert L0 > 0
        self.loss = loss.warping_flow_loss
        self.weight = weight
        self.L0 = L0
        self.default_return = None

    def __call__(self, i, image1, flow):
        """
        flow is from image1 to image0 
        """
        loss = self.default_return if i < self.L0 else self.weight * self.loss(
                self.image0, image1, flow)
        self.image0 = image1
        return loss


class voxel_warp_flow_loss():
    def __init__(self, weight=1.0):
        self.loss = loss.voxel_warping_flow_loss
        self.weight = weight

    def __call__(self, voxel, displacement, output_images=False):
        """
        Warp the voxel grid by the displacement map. Variance 
        of resulting image is loss
        """
        loss = self.loss(voxel, displacement, output_images)
        if output_images:
            loss = (self.weight * loss[0], loss[1])
        else:
            loss *= self.weight
        return loss


class flow_perceptual_loss():
    def __init__(self, weight=1.0, use_gpu=True):
        """
        Flow wrapper for perceptual_loss
        """
        self.loss = perceptual_loss(weight=1.0, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred, target):
        """
        pred and target are Tensors with shape N x 2 x H x W
        PerceptualLoss expects N x 3 x H x W.
        """
        dist_x = self.loss(pred[:, 0:1, :, :], target[:, 0:1, :, :], normalize=False)
        dist_y = self.loss(pred[:, 1:2, :, :], target[:, 1:2, :, :], normalize=False)
        return (dist_x + dist_y) / 2 * self.weight


class flow_l1_loss():
    def __init__(self, weight=1.0):
        self.loss = F.l1_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


# keep for compatibility
flow_loss = flow_l1_loss


class perceptual_loss():
    def __init__(self, weight=1.0, net='alex', use_gpu=True):
        """
        Wrapper for PerceptualSimilarity.models.PerceptualLoss
        """
        self.model = models.PerceptualLoss(net=net, use_gpu=use_gpu)
        self.weight = weight

    def __call__(self, pred, target, normalize=True):
        """
        pred and target are Tensors with shape N x C x H x W (C {1, 3})
        normalize scales images from [0, 1] to [-1, 1] (default: True)
        PerceptualLoss expects N x 3 x H x W.
        """
        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
        if target.shape[1] == 1:
            target = torch.cat([target, target, target], dim=1)
        dist = self.model.forward(pred, target, normalize=normalize)
        return self.weight * dist.mean()


class l2_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class psnr_loss():

    def __init__(self, weight=1.0, reduction='mean', toY=False):
        self.scale = 10 / np.log(10)
        self.weight = weight
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def __call__(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()





class temporal_consistency_loss():
    def __init__(self, weight=1.0, L0=1):
        assert L0 > 0
        self.loss = loss.temporal_consistency_loss
        self.weight = weight
        self.L0 = L0

    def __call__(self, i, image1, processed1, flow, output_images=False):
        """
        flow is from image1 to image0 
        """
        if i >= self.L0:
            loss = self.loss(self.image0, image1, self.processed0, processed1,
                             flow, output_images=output_images)
            if output_images:
                loss = (self.weight * loss[0], loss[1])
            else:
                loss *= self.weight
        else:
            loss = None
        self.image0 = image1
        self.processed0 = processed1
        return loss



class ssim_loss():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)



# duplicated code for compatibility with trainer
# it is a whole lot of them...
class ssim_loss_inf():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)


class ssim_loss_vis():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)


class l2_loss_inf():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class l2_loss_vis():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class ssim_loss_inf_rec():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)


class ssim_loss_vis_rec():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)


class l2_loss_inf_rec():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class l2_loss_vis_rec():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class ssim_loss_avpv():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)

class l2_loss_avpv():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)



class ssim_loss_aipi():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)

class l2_loss_aipi():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class ssim_loss_avpi():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)

class l2_loss_avpi():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class ssim_loss_aipv():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)

class l2_loss_aipv():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)

class ssim_loss_fuse_vis():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)

class l2_loss_fuse_vis():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)

class ssim_loss_fuse_inf():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)

class l2_loss_fuse_inf():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)

class max_mi_loss_vis():
    def __init__(self, weight=1.0):
        self.weight = weight
        
    def __call__(self, mi):
        return self.weight * -1.0 * mi

class max_mi_loss_inf():
    def __init__(self, weight=1.0):
        self.weight = weight
        
    def __call__(self, mi):
        return self.weight * -1.0 * mi

class min_mi_loss_vis_inf():
    def __init__(self, weight=1.0):
        self.weight = weight
        
    def __call__(self, mi):
        try:
            if mi.shape[0] > 1:
                mi = torch.sum(mi, 0)
        except:
            pass
        return self.weight * mi

class max_mi_loss_vis_inf():
    def __init__(self, weight=1.0):
        self.weight = weight
        
    def __call__(self, mi):
        try:
            if mi.shape[0] > 1:
                mi = torch.sum(mi, 0)
        except:
            pass
        return self.weight * mi
    

class l2_loss_inf1():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)

class ssim_loss_inf1():
    def __init__(self, weight=1.0):
        self.loss = msssim
        self.weight = weight

    def __call__(self, pred, target):
        ssim = self.loss(pred, target, normalize=True)
        return self.weight * (1.0 - ssim)

class max_cos_loss_vis():
    def __init__(self, weight=1.0):
        self.weight = weight
        
    def __call__(self, mi):
        return self.weight * mi

class max_cos_loss_inf():
    def __init__(self, weight=1.0):
        self.weight = weight
        
    def __call__(self, mi):
        return self.weight * mi

class infonce_loss_vis():
    def __init__(self, weight=1.0):
        self.loss = InfoNCE()
        self.weight = weight
        
    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)

class infonce_loss_inf():
    def __init__(self, weight=1.0):
        self.loss = InfoNCE()
        self.weight = weight
        
    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)


class club_min_mi_loss():
    def __init__(self, weight=1.0, CLUB_INTERVAL = 200):
        self.counter = 0
        self.weight = weight
        self.CLUB_INTERVAL = CLUB_INTERVAL
        
    def __call__(self, mi):
        self.counter += 1
        if self.counter > self.CLUB_INTERVAL:
            loss =  self.weight * mi
        else:
            loss = 0.0 * mi
        if self.counter >= 2*self.CLUB_INTERVAL:
            self.counter = 0
        scale = 0.5
        while True:
            if torch.abs(loss) > 0.50:
                loss = loss * scale
            else:
                break
        return loss

class club_learn_loss():
    def __init__(self, weight=1.0, CLUB_INTERVAL = 200):
        self.counter = 0
        self.weight = weight
        self.CLUB_INTERVAL = CLUB_INTERVAL
        
    def __call__(self, mi):
        self.counter += 1
        if self.counter <= self.CLUB_INTERVAL:
            loss =  self.weight * mi
        else:
            loss = 0.0 * mi
        if self.counter >= 2*self.CLUB_INTERVAL:
            self.counter = 0
        return loss


class seafusion_loss():
    def __init__(self, weight=1.0):
        self.loss = Fusionloss()
        self.weight = weight
        
    def __call__(self, image_vis,image_ir,image_fuse):
        return self.weight * self.loss(image_vis,image_ir,image_fuse)


