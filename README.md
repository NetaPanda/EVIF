Code for Event-based Visible and Infrared Fusion via Multi-task Collaboration (CVPR 2024)

Usage:

1. Download Data from Baidu Netdisk:

link: https://pan.baidu.com/s/1-KVYVTu0oxA_RzzwrylmFg?pwd=5rke 
extraction code: 5rke 

Note: Due to the seperate data generation process, there are two folders for the KAIST synthetic dataset:
The evt_rec_h5_dataset folder contains simulated raw events, VIS and LWIR frames and optical flows (calculated with mmflow)

The infrare_deblur_h5_dataset_with_vis_gt_blur_newnew_thin_copy contains blurry LWIR frames and corresponding sharp frames

A piece of advice: The dataset size is nearly 400GB. You may download only a subset of files and try the code.

We obtained the original KAIST dataset from https://github.com/SoonminHwang/rgbt-ped-detection/tree/master/data
Where the original dataset file was downloaded from

http://multispectral.kaist.ac.kr/pedestrian/data-kaist/videos.tar

and 

http://multispectral.kaist.ac.kr/pedestrian/data-kaist/annotations.tar

Since the above two links are not available now, we also provide the original data files with Baidu Netdisk for generating
synthetic dataset with other settings:

https://pan.baidu.com/s/1GW-dXjXBaubee2-MrPtpsA?pwd=k38d


2. Modify the paths in the dataset txt files to your own path


3. Train task 1-3, in each task's folder, run bash run_train.sh for training
Remember to check the config files and modify the path to your own path whenever necessary (e.g., the dataset path, the save path, and the pretrained model path in the config files)
For inference, run bash run_inference.sh

# Code for Event-based Visible and Infrared Fusion via Multi-task Collaboration (CVPR 2024)

## Usage

### 1. Download KAIST Synthetic Data from Baidu Netdisk

- **Link**: [https://pan.baidu.com/s/1-KVYVTu0oxA_RzzwrylmFg?pwd=5rke](https://pan.baidu.com/s/1-KVYVTu0oxA_RzzwrylmFg?pwd=5rke)  

Note: Due to the separate data generation process, the KAIST synthetic dataset is organized into two folders:

- **evt_rec_h5_dataset**: Contains simulated raw events, VIS and LWIR frames, and optical flows (calculated with [mmflow](https://github.com/open-mmlab/mmflow)).
- **infrarred_deblur_h5_dataset_with_vis_gt_blur_newnew_thin_copy**: Contains blurry LWIR frames and corresponding sharp frames.

**Advice**: The dataset is approximately 400GB in size. You can download a subset of files first to run the code.

We obtained the original KAIST dataset from:  
[https://github.com/SoonminHwang/rgbt-ped-detection/tree/master/data](https://github.com/SoonminHwang/rgbt-ped-detection/tree/master/data)

The original dataset files can be downloaded from:  
- [http://multispectral.kaist.ac.kr/pedestrian/data-kaist/videos.tar](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/videos.tar)  
- [http://multispectral.kaist.ac.kr/pedestrian/data-kaist/annotations.tar](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/annotations.tar)

Since these links are no longer available, we are also providing the original data files via Baidu Netdisk for generating synthetic datasets with other settings:

- **Baidu Netdisk Link**: [https://pan.baidu.com/s/1GW-dXjXBaubee2-MrPtpsA?pwd=k38d](https://pan.baidu.com/s/1GW-dXjXBaubee2-MrPtpsA?pwd=k38d)

### 2. Modify Dataset Paths

Update the paths in the dataset `.txt` files to match your local environment.

### 3. Training Tasks

To train tasks 1-3, navigate to the corresponding task's folder and run:

```bash
bash run_train.sh

Make sure to check and modify the configuration files to set the correct paths for the dataset, save directories, and any pretrained models if needed.
The pretrained model needed for task 1 (update_reconstruction_model.pth) can be downloaded from [event_cnn_minimal](https://github.com/TimoStoff/event_cnn_minimal)

For inference, run:

```bash
bash run_inference.sh 


Some quick ablation study results (task 3 trained run_train.sh, get fused frames with run_inference.sh, then tested with [VIFB](https://github.com/xingchenzhang/VIFB)):

| Metric               | Task 3 w/o MI optimization | Task 3 w/ only MI minimization | Task 3 w/ MI min-max |
|----------------------|----------------------------|--------------------------------|-----------------------|
| Cross_entropy ↓      | 1.3952                     | 1.3930                         | 1.3167                |
| Entropy ↑            | 7.2545                     | 7.2461                         | 7.2514                |
| Mutinf ↑             | 2.6546                     | 2.7193                         | 2.7902                |
| Psnr ↑               | 58.1774                    | 58.1810                        | 58.3648               |
| Avg_gradient ↑       | 3.0795                     | 3.1351                         | 3.1222                |
| Qabf ↑               | 0.6303                     | 0.6304                         | 0.6443                |
| Variance ↑           | 47.5371                    | 47.1482                        | 47.4490               |
| Spatial_frequency ↑  | 8.1453                     | 8.4109                         | 8.2886                |
| Rmse ↓               | 0.1038                     | 0.1038                         | 0.1002                |
| Ssim ↑               | 1.4413                     | 1.4355                         | 1.4429                |
| Qcb ↑                | 0.4489                     | 0.4511                         | 0.4639                |
| Qcv ↓                | 365.0677                   | 371.0641                       | 356.5801              |




This code is based on [event_cnn_minimal](https://github.com/TimoStoff/event_cnn_minimal) and [EFNet](https://github.com/AHupuJR/EFNet), we also take inspiration from other works such as [SeAFusion](https://github.com/Linfeng-Tang/SeAFusion) and [YDTR](https://github.com/tthinking/YDTR), thanks to these great works.

