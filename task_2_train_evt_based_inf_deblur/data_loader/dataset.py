from torch.utils.data import Dataset
import numpy as np
import random
import torch
import h5py
import os
# local modules
from utils.data_augmentation import *
from utils.data import data_sources
from events_contrast_maximization.utils.event_utils import events_to_voxel_torch, \
    events_to_neg_pos_voxel_torch, binary_search_torch_tensor, events_to_image_torch, \
    binary_search_h5_dset, get_hot_event_mask, save_image
from utils.util import read_json, write_json
import traceback


class BaseVoxelDataset(Dataset):
    """
    Dataloader for joint task of event reconstruction, event-based infrared deblur, and infrare-vis fusion
    For each index, returns a dict containing:
        * blurry_frame is a H x W tensor containing blurry infrared frame
        * blurry_events is a C x H x W tensor containing the voxel grid during the exposure time of the blurry_frame
          C = num_bins * 2 since we gonna seperate pos and neg event voxels
        * history_events is a K*C x H x W tensor containing K voxel grid that corresponding to the K exposure windows before the mid timestamp of the blurry_frame
        *   these K voxel grid are used to feed into the recurrent evt rec network so that at last it output exactly the vis recon frame at the sharp middle infrared frame's ts
        * sharp_frame is a H x W tensor that is the ground truth of the deblurred blurry_frame
        * sharp_vis_frame is a H x W tensor that is the ground truth visible frame that at the same timestamp (ts) as sharp_frame
    Subclasses must implement:
        - get_blurry_frame(index) method which retrieves the blurry frame at index i
        - get_sharp_frame(index) method which retrieves the sharp frame at index i
        - get_sharp_vis_frame(index) method which retrieves the sharp vis frame at index i
        - get_blurry_frame_ts(index) method which retrieves the blurry frame's exposure start/mid/end ts, mid ts equal to the sharp frames's ts
        - get_events(idx0, idx1) method which gets the events between idx0 and idx1
            (in format xs, ys, ts, ps, where each is a np array
            of x, y positions, timestamps and polarities respectively)
        - load_data() initialize the data loading method and ensure the following
            members are filled:
            sensor_resolution - the sensor resolution
            num_frames - the number of frames
            dt - single history window size
        - find_ts_index(timestamp) given a timestamp, find the index of
            the corresponding event

    Parameters:
        data_path_evt_rec Path to the file containing the event reconstruction data (containing events, vis images(we dont use this), and maybe flows (we dont use this))
        data_path_deblur Path to the file containing the deblur data (containing blurry/sharp infrared images, sharp vis images (we use this), and blurry frame ts)
        transforms Dict containing the desired augmentations
        sensor_resolution The size of the image sensor from which the events originate
        num_bins The number of bins desired in the voxel grid
        history_size The number of K for returning history events
        skip_size Skip how many indexes, since KAIST data usualy have ~100 frames (~15 blurry frames) still images
        voxel_method Which method should be used to form the voxels.
            Currently supports:
            * "k_events" (new voxels are formed every k events)
            * "t_seconds" (new voxels are formed every t seconds)
            * "between_frames" (all events between frames are taken, requires frames to exist)
            A sliding window width must be given for k_events and t_seconds,
            which determines overlap (no overlap if set to 0). Eg:
            method={'method':'k_events', 'k':10000, 'sliding_window_w':100}
            method={'method':'t_events', 't':0.5, 'sliding_window_t':0.1}
            method={'method':'between_frames'}
            Default is 'between_frames'.
    """

    def get_blurry_frame(self, index):
        """
        Get blurry infrared frame at index
        """
        raise NotImplementedError

    def get_sharp_frame(self, index):
        """
        Get sharp gt infrared frame at index
        """
        raise NotImplementedError

    def get_sharp_vis_frame(self, index):
        """
        Get sharp vis frame at index
        """
        raise NotImplementedError

    def get_blurry_frame_ts(self, index):
        """
        Get the blurry frame's exposure start/mid/end ts at index
        """
        raise NotImplementedError


    def get_events(self, idx0, idx1):
        """
        Get events between idx0, idx1
        """
        raise NotImplementedError

    def load_data(self, data_path_evt_rec, data_path_deblur):
        """
        Perform initialization tasks and ensure essential members are populated.
        Required members are:
            members are filled:
            self.sensor_resolution - the sensor resolution
            self.num_frames - the number of frames
        """
        raise NotImplementedError

    def find_ts_index(self, timestamp):
        """
        Given a timestamp, find the event index
        """
        raise NotImplementedError


    def __init__(self, data_path_evt_rec, data_path_deblur, transforms={}, sensor_resolution=None, num_bins=5, history_size = 20, dt=0.05, skip_size = 15, voxel_method=None, combined_voxel_channels=True, noise_kwargs=None, hot_pixel_kwargs=None, crop_size=None):
        """
        self.transform applies to event voxels, frames and flow.
        self.vox_transform applies to event voxels only.
        """

        self.noise_kwargs = noise_kwargs
        self.hot_pixel_kwargs = hot_pixel_kwargs
        self.num_bins = num_bins
        self.data_path_evt_rec = data_path_evt_rec
        self.data_path_deblur = data_path_deblur
        # must be true
        self.combined_voxel_channels = combined_voxel_channels
        self.sensor_resolution = sensor_resolution
        self.channels = self.num_bins if combined_voxel_channels else self.num_bins*2
        self.sensor_resolution, self.num_frames, self.dt = None, None, None
        self.dt = dt # in seconds
        self.skip_size = skip_size
        self.history_size = history_size
        self.load_data(data_path_evt_rec, data_path_deblur)
        if crop_size is None:
            self.crop_size = self.sensor_resolution
        else:
            self.crop_size = [crop_size, crop_size]


        if self.sensor_resolution is None or self.num_frames is None or self.dt is None or self.crop_size is None:
            raise Exception("Dataloader failed to intialize all required members ({})".format(self.data_path_evt_rec))

        self.num_pixels = self.sensor_resolution[0] * self.sensor_resolution[1]
        print(self.sensor_resolution)

        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.set_voxel_method(voxel_method)
        self.hot_events_mask = np.ones([self.channels, *self.sensor_resolution])
        self.hot_events_mask = torch.from_numpy(self.hot_events_mask).float()

        if 'LegacyNorm' in transforms.keys() and 'RobustNorm' in transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
                del (transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

        transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)
        if not self.normalize_voxels:
            self.vox_transform = self.transform

        self.voxel_cache = {}
        self.his_voxel_cache = {}

    def __getitem__(self, index, seed=None):
        """
        Get data at index.
            :param index: index of data
            :param seed: random seed for data augmentation
        """
        index = index + self.skip_size
        assert 0 <= index < self.__len__() + self.skip_size, "index {} out of bounds (0 <= x < {})".format(index, self.__len__())
        seed = random.randint(0, 2 ** 32) if seed is None else seed
        random.seed(seed)
        # get blurry frame exposure window events
        idx0, idx1 = self.get_event_indices(index)
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        ts = torch.from_numpy(ts.astype(np.float32))
        ps = torch.from_numpy(ps.astype(np.float32))

        # just pre-crop all things
        H = self.sensor_resolution[0]
        W = self.sensor_resolution[1]
        h_start = random.randint(0, H-self.crop_size[0])
        h_end   = h_start + self.crop_size[0]
        w_start = random.randint(0, W-self.crop_size[1])
        w_end   = w_start + self.crop_size[1]
        # filter events
        cond1 = (xs >= w_start) 
        cond2 = (xs < w_end) 
        cond3 = (ys >= h_start) 
        cond4 = (ys < h_end) 
        index12 = torch.logical_and(cond1,cond2)
        index34 = torch.logical_and(cond3,cond4)
        indexxy = torch.logical_and(index12,index34)
        indices = indexxy.nonzero().squeeze()
        xs = torch.index_select(xs, 0, indices) - w_start
        ys = torch.index_select(ys, 0, indices) - h_start
        ts = torch.index_select(ts, 0, indices)
        ps = torch.index_select(ps, 0, indices)

        xs = xs.detach().cpu().numpy()
        ys = ys.detach().cpu().numpy()
        ts = ts.detach().cpu().numpy()
        ps = ps.detach().cpu().numpy()
        # filter end

        try:
            ts_0, ts_k  = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0

        if index in self.voxel_cache:
            voxel = self.voxel_cache[index]
        else:
            if len(xs) < 3:
                voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
            else:
                if index in self.voxel_cache:
                    voxel = self.voxel_cache[index]
                else:
                    if False:
                        pass
                    else:
                        xs1 = torch.from_numpy(xs.astype(np.float32))
                        ys1 = torch.from_numpy(ys.astype(np.float32))
                        ts1 = torch.from_numpy((ts-ts_0).astype(np.float32))
                        ps1 = torch.from_numpy(ps.astype(np.float32))
                        hot_events_mask=self.hot_events_mask[:,h_start:h_end,w_start:w_end]
                        voxel = self.get_voxel_grid(xs1, ys1, ts1, ps1, combined_voxel_channels=self.combined_voxel_channels, sensor_size=self.crop_size, hot_events_mask=hot_events_mask)
                        voxel_np = voxel.detach().cpu().numpy()

            #self.voxel_cache[index] = voxel
            #print('Voxel cache ', index)

        voxel = self.transform_voxel(voxel, seed).float()
        dt = ts_k - ts_0
        if dt == 0:
            dt = np.array(0.0)

        # get history events
        # divide blurry frame exposure window events into K-1 segments
        # K is how many frames we form a blurry frame
        # eg, if we use t1,t1+dt,t2+dt,...t7  to form a blurry frame, then we return K-1 windows of [ [t1,t1+dt], [t1+dt,t1+2*dt]....[t7-dt, t7] ], 

        # first we gonna divide the events temporally with K-1 segments
        K = 7
        this_frame_ts = self.frame_ts[index]
        this_frame_ts_start = this_frame_ts[0]
        this_frame_ts_mid = this_frame_ts[1]
        this_frame_ts_end = this_frame_ts[2]
        sub_ts = np.linspace(this_frame_ts_start, this_frame_ts_end, K)
        history_voxels = []
        for i in range(K-1):
            sub_t_start = sub_ts[i]
            sub_t_end = sub_ts[i+1]
            # find sub interval events in xs, ys, ts, ps we read above, ts is in numpy but numpy and torch tensor operates the same
            sub_idx_start = binary_search_torch_tensor(ts, 0, len(ts)-1, sub_t_start)
            sub_idx_end = binary_search_torch_tensor(ts, 0, len(ts)-1, sub_t_end)
            his_xs = xs[sub_idx_start:sub_idx_end]
            his_ys = ys[sub_idx_start:sub_idx_end]
            his_ts = ts[sub_idx_start:sub_idx_end]
            his_ps = ps[sub_idx_start:sub_idx_end]
            # get voxel
            try:
                his_ts_0, his_ts_k  = his_ts[0], his_ts[-1]
            except:
                his_ts_0, his_ts_k = 0, 0
            if len(his_xs) < 3:
                his_voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
            else:
                his_xs = torch.from_numpy(his_xs.astype(np.float32))
                his_ys = torch.from_numpy(his_ys.astype(np.float32))
                his_ts = torch.from_numpy((his_ts-his_ts_0).astype(np.float32))
                his_ps = torch.from_numpy(his_ps.astype(np.float32))
                hot_events_mask=self.hot_events_mask[:,h_start:h_end,w_start:w_end]
                his_voxel = self.get_voxel_grid(his_xs, his_ys, his_ts, his_ps, combined_voxel_channels=self.combined_voxel_channels, sensor_size=self.crop_size, hot_events_mask=hot_events_mask)

            history_voxels.append(his_voxel)

 
        # turn history_voxels to torch
        # [K, C, H, W]
        history_voxels = torch.stack(history_voxels, 0)

        processed_his_voxels = []
        for i in range(history_voxels.shape[0]):
            processed_his_voxels.append(self.transform_voxel(history_voxels[i,...], seed).float())
        history_voxels = torch.stack(processed_his_voxels, 0)


        # get previous M frame's history voxels
        M = 1
        # this is used to warmup the recurrent encoder
        prev_idx0, prev_idx1 = self.get_event_indices(index-M)
        # get three windows together, from the index-3 frame's start idx to this frame's start idx
        prev_xs, prev_ys, prev_ts, prev_ps = self.get_events(prev_idx0, idx0)

        # crop events
        prev_xs = torch.from_numpy(prev_xs.astype(np.float32))
        prev_ys = torch.from_numpy(prev_ys.astype(np.float32))
        prev_ts = torch.from_numpy(prev_ts.astype(np.float32))
        prev_ps = torch.from_numpy(prev_ps.astype(np.float32))

        # filter evenprev_ts
        cond1 = (prev_xs >= w_start) 
        cond2 = (prev_xs < w_end) 
        cond3 = (prev_ys >= h_start) 
        cond4 = (prev_ys < h_end) 
        index12 = torch.logical_and(cond1,cond2)
        index34 = torch.logical_and(cond3,cond4)
        indexxy = torch.logical_and(index12,index34)
        indices = indexxy.nonzero().squeeze()
        prev_xs = torch.index_select(prev_xs, 0, indices) - w_start
        prev_ys = torch.index_select(prev_ys, 0, indices) - h_start
        prev_ts = torch.index_select(prev_ts, 0, indices)
        prev_ps = torch.index_select(prev_ps, 0, indices)

        prev_xs = prev_xs.detach().cpu().numpy()
        prev_ys = prev_ys.detach().cpu().numpy()
        prev_ts = prev_ts.detach().cpu().numpy()
        prev_ps = prev_ps.detach().cpu().numpy()
        # filter end


        # first we gonna divide the events temporally with K*M segments
        # copy the code above, variable name might not correct
        W = K * M
        this_frame_ts = self.frame_ts[index-M]
        this_frame_ts_start = this_frame_ts[0]
        this_frame_ts = self.frame_ts[index]
        this_frame_ts_end = this_frame_ts[0]
        sub_ts = np.linspace(this_frame_ts_start, this_frame_ts_end, W+1)
        prev_history_voxels = []
        for i in range(W):
            sub_t_start = sub_ts[i]
            sub_t_end = sub_ts[i+1]
            # find sub interval events in xs, ys, ts, ps we read above, ts is in numpy but numpy and torch tensor operates the same
            sub_idx_start = binary_search_torch_tensor(prev_ts, 0, len(prev_ts)-1, sub_t_start)
            sub_idx_end = binary_search_torch_tensor(prev_ts, 0, len(prev_ts)-1, sub_t_end)
            his_xs = prev_xs[sub_idx_start:sub_idx_end]
            his_ys = prev_ys[sub_idx_start:sub_idx_end]
            his_ts = prev_ts[sub_idx_start:sub_idx_end]
            his_ps = prev_ps[sub_idx_start:sub_idx_end]
            # get voxel
            try:
                his_ts_0, his_ts_k  = his_ts[0], his_ts[-1]
            except:
                his_ts_0, his_ts_k = 0, 0
            if len(his_xs) < 3:
                his_voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
            else:
                his_xs = torch.from_numpy(his_xs.astype(np.float32))
                his_ys = torch.from_numpy(his_ys.astype(np.float32))
                his_ts = torch.from_numpy((his_ts-his_ts_0).astype(np.float32))
                his_ps = torch.from_numpy(his_ps.astype(np.float32))
                hot_events_mask=self.hot_events_mask[:,h_start:h_end,w_start:w_end]
                his_voxel = self.get_voxel_grid(his_xs, his_ys, his_ts, his_ps, combined_voxel_channels=self.combined_voxel_channels, sensor_size=self.crop_size, hot_events_mask=hot_events_mask)
                #if torch.sum(torch.isnan(his_voxel)).item() > 0:
                #    import ipdb
                #    ipdb.set_trace()

            prev_history_voxels.append(his_voxel)

 
        # turn history_voxels to torch
        # [K, C, H, W]
        prev_history_voxels = torch.stack(prev_history_voxels, 0)

        processed_prev_his_voxels = []
        for i in range(prev_history_voxels.shape[0]):
            processed_prev_his_voxels.append(self.transform_voxel(prev_history_voxels[i,...], seed).float())
        prev_history_voxels = torch.stack(processed_prev_his_voxels, 0)


        if self.voxel_method['method'] == 'between_frames':
            blurry_frame = self.get_blurry_frame(index)
            blurry_frame = blurry_frame[h_start:h_end,w_start:w_end] 
            blurry_frame = self.transform_frame(blurry_frame, seed)
            sharp_frame = self.get_sharp_frame(index)
            sharp_frame = sharp_frame[h_start:h_end,w_start:w_end] 
            sharp_frame = self.transform_frame(sharp_frame, seed)
            sharp_vis_frame = self.get_sharp_vis_frame(index)
            sharp_vis_frame = sharp_vis_frame[h_start:h_end,w_start:w_end] 
            sharp_vis_frame = self.transform_frame(sharp_vis_frame, seed)
            item = {'blurry_frame': blurry_frame,
                    'blurry_events': voxel,
                    'history_events': history_voxels,
                    'prev_history_events': prev_history_voxels,
                    'sharp_frame': sharp_frame,
                    'sharp_vis_frame': sharp_vis_frame,
                    'num_bins': torch.tensor(self.dt, dtype=torch.int64),
                    'data_source_idx': torch.tensor(-1, dtype=torch.int64),
                    'dt': torch.tensor(self.dt, dtype=torch.float64)}
            if self.noise_kwargs:
                item['blurry_events'] = add_noise_to_voxel(item['blurry_events'], **self.noise_kwargs)
                for i in range(len(item['history_events'])):
                    item['history_events'][i,...] = add_noise_to_voxel(item['history_events'][i,...], **self.noise_kwargs)
                for i in range(len(item['prev_history_events'])):
                    item['prev_history_events'][i,...] = add_noise_to_voxel(item['prev_history_events'][i,...], **self.noise_kwargs)

            if self.hot_pixel_kwargs:
                item = add_hot_pixels_to_voxels(item, **self.hot_pixel_kwargs)

        else:
            print("Not between")
            raise NotImplementedError

        return item

    def compute_frame_indices(self):
        """
        For each frame, find the start and end indices of the
        time synchronized events
        """
        raise NotImplementedError

    def set_voxel_method(self, voxel_method):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        """
        self.voxel_method = voxel_method
        if self.voxel_method['method'] == 'between_frames':
            self.length = self.num_frames - self.skip_size
            self.event_indices, self.frame_ts = self.compute_frame_indices()
        else:
            raise Exception("Invalid voxel forming method chosen ({})".format(self.voxel_method))
        if self.length == 0:
            raise Exception("Current voxel generation parameters lead to sequence length of zero")

    def __len__(self):
        return self.length

    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return idx0, idx1

    def get_history_event_indices(self, index):
        """
        Get start and end indices of events at index
        """
        idxs = self.history_event_indices[index]
        return idxs




    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.num_bins, *self.crop_size)
        else:
            size = (2*self.num_bins, *self.crop_size)
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True, sensor_size=None, hot_events_mask=None):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """

        if sensor_size is None:
            sensor_size = self.sensor_resolution
        if hot_events_mask is None:
            hot_events_mask = self.hot_events_mask

        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=sensor_size)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                                       sensor_size=sensor_size)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        voxel_grid = voxel_grid*hot_events_mask 

        return voxel_grid

    def transform_frame(self, frame, seed):
        """
        Augment frame and turn into tensor
        """
        frame = torch.from_numpy(frame).float().unsqueeze(0) / 255
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed):
        """
        Augment voxel and turn into tensor
        """
        if self.vox_transform:
            random.seed(seed)
            voxel = self.vox_transform(voxel)
        return voxel

    def transform_flow(self, flow, seed):
        """
        Augment flow and turn into tensor
        """
        flow = torch.from_numpy(flow)  # should end up [2 x H x W]
        if self.transform:
            random.seed(seed)
            flow = self.transform(flow, is_flow=True)
        return flow


class DynamicH5Dataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the Monash University HDF5 events format
    (see https://github.com/TimoStoff/event_utils for code to convert datasets)
    """

    def get_blurry_frame(self, index):
        frame = self.h5_file_deblur['blur_images']['image{:09d}'.format(index)][:]
        # frame is expected to be [H,W], not [H,W,1]
        if frame.ndim == 3:
            frame = np.squeeze(frame)
        return frame

    def get_sharp_frame(self, index):
        frame = self.h5_file_deblur['gt_images']['image{:09d}'.format(index)][:]
        # frame is expected to be [H,W], not [H,W,1]
        if frame.ndim == 3:
            frame = np.squeeze(frame)
        return frame

    def get_sharp_vis_frame(self, index):
        frame = self.h5_file_deblur['vis_images']['image{:09d}'.format(index)][:]
        # frame is expected to be [H,W], not [H,W,1]
        if frame.ndim == 3:
            frame = np.squeeze(frame)
        return frame


    def get_flow(self, index):
        return self.h5_file['flow']['flow{:09d}'.format(index)][:]

    def get_events(self, idx0, idx1):
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0
        return xs, ys, ts, ps

    def load_data(self, data_path_evt_rec, data_path_deblur):
        try:
            self.h5_file = h5py.File(data_path_evt_rec, 'r')
            # use r+ since we might right index into it
            # since index computation is too slow to be on the fly
            #self.h5_file_deblur = h5py.File(data_path_deblur, 'r+')
            # once we passed through the whole dataset once and wrote what we want, switch to this line
            # to avoid blockage of multiple reading
            self.h5_file_deblur = h5py.File(data_path_deblur, 'r')
        except OSError as err:
            traceback.print_exc()
            print("Couldn't open {}: {}".format(data_path_evt_rec, err))

        if self.sensor_resolution is None:
            self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]
        print("sensor resolution = {}".format(self.sensor_resolution))
        self.t0 = self.h5_file['events/ts'][0]
        self.tk = self.h5_file['events/ts'][-1]
        self.num_events = self.h5_file.attrs["num_events"]
        self.num_frames = self.h5_file_deblur.attrs["num_imgs"]


    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx


    def get_blurry_frame_ts(self, index):
        start_ts = self.h5_file_deblur['blur_images/image{:09d}'.format(index)].attrs['timestamp_start']
        mid_ts = self.h5_file_deblur['blur_images/image{:09d}'.format(index)].attrs['timestamp_mid']
        end_ts = self.h5_file_deblur['blur_images/image{:09d}'.format(index)].attrs['timestamp_end']
        return start_ts, mid_ts, end_ts


    def compute_frame_indices(self):
        frame_ts = []
        frame_indices = [] # index of the exposure range events
        frame_history_indices = [] # index of K windows before and within the exposure range, each is [start_idx, end_idx], K is how many frames we form a blurry frame
        # eg, if we use t1,t1+dt,t2+dt,...t7  to form a blurry frame, then we return K windows of [ [t1-dt,t1], [t1,t1+dt], [t1+dt,t1+2*dt]....[t7-dt, t7] ], 
        start_idx = 0
        count = 0
        wrote_h5 = False
        for img_name in self.h5_file_deblur['blur_images']:
            start_ts = self.h5_file_deblur['blur_images/{}'.format(img_name)].attrs['timestamp_start']
            mid_ts = self.h5_file_deblur['blur_images/{}'.format(img_name)].attrs['timestamp_mid']
            end_ts = self.h5_file_deblur['blur_images/{}'.format(img_name)].attrs['timestamp_end']
            if 'start_idx' in self.h5_file_deblur['blur_images/{}'.format(img_name)].attrs.keys():
                start_idx = self.h5_file_deblur['blur_images/{}'.format(img_name)].attrs['start_idx']
                mid_idx = self.h5_file_deblur['blur_images/{}'.format(img_name)].attrs['mid_idx']
                end_idx = self.h5_file_deblur['blur_images/{}'.format(img_name)].attrs['end_idx']
            else:
                start_idx = self.find_ts_index(start_ts)
                mid_idx = self.find_ts_index(mid_ts)
                end_idx = self.find_ts_index(end_ts)
                self.h5_file_deblur['blur_images/{}'.format(img_name)].attrs['start_idx'] = start_idx
                self.h5_file_deblur['blur_images/{}'.format(img_name)].attrs['mid_idx'] = mid_idx
                self.h5_file_deblur['blur_images/{}'.format(img_name)].attrs['end_idx'] = end_idx
                wrote_h5 = True

            frame_indices.append([start_idx, end_idx])
            frame_ts.append([start_ts, mid_ts, end_ts])


            count = count + 1
            if count % 100 == 0:
                print(count)
                if wrote_h5:
                    print('Updated h5 file to add indices, after read through the whole dataset, you should change')
                    print('the r+ in h5py.File(data_path_deblur, \'r+\') to r (dataset.py), incase multiple programs needs to read the same h5 file (r+ will block it)')
                          

        return frame_indices, frame_ts


