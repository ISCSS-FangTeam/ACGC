# Improved Pseudomasks Generation for Weakly Supervised Building Extraction From High-Resolution Remote Sensing Imagery

The implementation of Improved Pseudomasks Generation for Weakly Supervised Building Extraction From High-Resolution Remote Sensing Imagery, Fang Fang, Daoyuan Zheng, IJSTAR 2022. [[paper](https://ieeexplore.ieee.org/abstract/document/9684996/)]

# Installation

- We kindly refer to the offical implementation of [IRN ](https://github.com/jiwoon-ahn/irn) and  [AdvCAM ](https://github.com/jbeomlee93/AdvCAM).
- This repository is tested on Ubuntu 18.04, with Python 3.6, PyTorch 1.4, pydensecrf, scipy, chaniercv, imageio, and opencv-python.
## Usage

#### Step 1. 上传数据和标签(上传路径Dataset/potsdam/)

- JPEGImages:上传building类别和non_buidling类别的影像；
- SegmentationClassAug:上传building和non_buidling类别影像对应的mask标签；
- 生成cls_labels.npy:运行write_numpy.py文件，生成cls_labels.npy


#### Step 2. 训练分类网络，生成权重文件

```
python run_sample.py --train_cam_pass True \
					 --cam_batch_size 32 \
					 --cam_num_epoches 40 \
					 --voc12_root Dataset/potsdam\
					 --train_list cdy12/train_aug.txt \
					 --cam_learning_rate 0.001 --cam_weights_name sess/res50_cam_cdy.pth
注意注意：voc12_root 和 train_list 参数注意修改；
```

- Pre-trained model used in this paper: [Download](https://drive.google.com/file/d/1G0UkgjA4bndGBw2YFCrBpv71M5bj86qf/view?usp=sharing).
- You can also train your own classifiers following [IRN](https://github.com/jiwoon-ahn/irn).


#### Step 3. 生成类激活图
```
python obtain_CAM_masking.py --adv_iter 2 --AD_coeff 7 --AD_stepsize 0.08 --score_th 0.6 \
							   --voc12_root Dataset/potsdam  \
							   --train_list cdy12/train_aug.txt \
							   --cam_weights_name sess/res50_cam_cdy.pth \
							   --cam_out_dir result/cam_adv_mask_cdy
注意注意：voc12_root 和 train_list 参数注意修改  adv_iter为迭代次数，遥感影像建筑物提取任务不会设置太大；
```

#### Step 4. 评估类激活图精度

```
python run_sample.py --eval_cam_pass True --cam_out_dir result/cam_adv_mask_cdy
注意注意：可修改run_sample.py文件中第111行阈值，以评估最优的类激活图；还需注意修改step/eval_cam.py文件中的路径;
```

#### Step 5. 生成IRlabel，供第五步训练

```
python run_sample.py --cam_to_ir_label_pass True --conf_fg_thres 0.5 --conf_bg_thres 0.4 \
					 --cam_out_dir result/cam_adv_mask_cdy \
					 --voc12_root Dataset/potsdam \
					 --train_list cdy12/train_aug.txt \
					 --ir_label_out_dir result/ir_label_cdy 
注意注意：conf_fg_thres和conf_bg_thres主要根据第三步中最优的阈值来确定；
```

#### Step 6. 训练门控卷积模块

```
python run_sample.py --train_irn_pass True --irn_batch_size 32 --irn_crop_size 256 --irn_num_epoches 10 \
					 --irn_weights_name sess/res50_irn_cdy.pth \
					 --voc12_root Dataset/potsdam  \
					 --train_list cdy12/train_aug_building.txt \
					 --ir_label_out_dir result/ir_label_cdy \
					 --infer_list cdy12/train_aug_building.txt \
					 --irn_learning_rate 0.1
注意注意：训练门控卷积模块 见net/resnet50_irn.py文件
```

#### Step 7. 生成分割标签并评估

```
python run_sample.py --make_sem_seg_pass True --eval_sem_seg_pass True --sem_seg_bg_thres 0.4 \
					 --cam_out_dir result/cam_adv_mask_cdy \
					 --sem_seg_out_dir result/sem_seg_cdy \
					 --irn_weights_name sess/res50_irn_cdy.pth \
					 --infer_list cdy12/train_aug_building.txt \
					 --voc12_root Dataset/potsdam
注意注意：sem_seg_bg_thres与之前步骤保持一致即可；还需注意修改step/eval_sem_seg.py文件中的路径
```


## Acknowledgment
This code is heavily borrowed from [IRN](https://github.com/jiwoon-ahn/irn) and [AdvCAM ](https://github.com/jbeomlee93/AdvCAM), thanks!