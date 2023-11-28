**HCSD-Net: Single Image Desnowing with Color Space Transformation （Accepted byProceeding of the 31st ACM International Conference on Multimedia）**

**Setup and environment**

To generate the restored result you need:

1. Python 3.7
2. CPU or NVIDIA GPU + CUDA CuDNN
3. Pytorch 1.8.0
4. python-opencv

**Testing**

We trained our model in four snow removal datasets, including Snow100K, CSD, SnowKitti2012, and SnowCityScapes.

Please replace weights_dir, data_dir, and results_dir in test.py, and put your test_dir in data_dir.

**Pre-trained model**

It can be downloaded from: https://pan.baidu.com/s/1jPHdKIH-tB188B24t5Bm1g.

Extract code: HCSD .

**Training**

You can train your own datasets by train.py. Please replace your own datasets_dir in it.

**Citations**

@inproceedings{10.1145/3581783.3613789,\
author = {Zhang, Ting and Jiang, Nanfeng and Wu, Hongxin and Zhang, Keke and Niu, Yuzhen and Zhao, Tiesong},\
title = {HCSD-Net: Single Image Desnowing with Color Space Transformation},\
year = {2023},\
doi = {10.1145/3581783.3613789},\
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},\
pages = {8125–8133},\
numpages = {9},\
series = {MM '23}\
}





