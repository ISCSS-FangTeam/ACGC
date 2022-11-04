import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from numpy.linalg import lstsq
from scipy.linalg import orth
import voc12.dataloader
from misc import torchutils, imutils
import cv2
from gradCAM import GradCAM

cudnn.enabled = True

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
parser.add_argument("--voc12_root", default='Dataset/VOC2012_SEG_AUG/', type=str,
                    help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
# Dataset
parser.add_argument("--train_list", default="voc12/train.txt", type=str)
parser.add_argument("--val_list", default="voc12/val.txt", type=str)
parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
                    help="voc12/train_aug.txt to train a fully supervised model, "
                         "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
parser.add_argument("--chainer_eval_set", default="train", type=str)

# Class Activation Map
parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
parser.add_argument("--cam_crop_size", default=512, type=int)
parser.add_argument("--cam_batch_size", default=2, type=int) # original: 16
parser.add_argument("--cam_num_epoches", default=5, type=int)
parser.add_argument("--cam_learning_rate", default=0.1, type=float)
parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
parser.add_argument("--cam_eval_thres", default=0.15, type=float)
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                    help="Multi-scale inferences")

parser.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
parser.add_argument("--target_layer", default="stage4")
parser.add_argument("--cam_out_dir", default="result/cam_adv_mask", type=str)
parser.add_argument("--adv_iter", default=27, type=int)
parser.add_argument("--AD_coeff", default=7, type=int)
parser.add_argument("--AD_stepsize", default=0.08, type=float)
parser.add_argument("--score_th", default=0.5, type=float)

args = parser.parse_args()
torch.set_num_threads(1)
if not os.path.exists(args.cam_out_dir):
    os.makedirs(args.cam_out_dir)


def adv_climb(image, epsilon, data_grad):
    sign_data_grad = data_grad / (torch.max(torch.abs(data_grad))+1e-12)
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, image.min().data.cpu().float(), image.max().data.cpu().float()) # min, max from data normalization
    return perturbed_image

def add_discriminative(expanded_mask, regions, score_th):
    region_ = regions / regions.max()
    expanded_mask[region_>score_th]=1
    return expanded_mask

def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=True)
    cam_sizes = [[], [], [], []] # scale 0,1,2,3
    with cuda.device(process_id):
        model.cuda()
        gcam = GradCAM(model=model, candidate_layers=[args.target_layer])  # stage4
        for iter, pack in enumerate(data_loader):
            # print(pack['name'])  # ['top_mosaic_09cm_area11_0_10']
            img_name = pack['name'][0] 
            # if os.path.exists(os.path.join(args.cam_out_dir, img_name + '.npy')):
            #     continue
            # if img_name != "top_mosaic_09cm_area1_4_3":
            #     continue
            size = pack['size']
            # print(size)  # [tensor([256]), tensor([256])]
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            # print(strided_up_size)
            outputs_cam = []
            n_classes = len(list(torch.nonzero(pack['label'][0])[:, 0]))
            if n_classes == 0:
                print('类别为0')
                continue
            # print(n_classes)
            
            for s_count, size_idx in enumerate([1, 0, 2, 3]):
                orig_img = pack['img'][size_idx].clone()
                for c_idx, c in enumerate(list(torch.nonzero(pack['label'][0])[:, 0])):
                    pack['img'][size_idx] = orig_img
                    # print(len(pack['img']))  # 4
                    # print(pack['img'][size_idx].shape)  # torch.Size([1, 2, 3, 128, 128])
                    img_single = pack['img'][size_idx].detach()[0]  # [:, 1]: flip
                    # 注意img_single有两个 一个是翻转前 一个是翻转后
                    # print('有类别，进来了:%s' % c)
                    if size_idx != 1:
                        total_adv_iter = args.adv_iter
                    else:
                        if args.adv_iter > 10:
                            total_adv_iter = args.adv_iter // 2  # 整除
                            mul_for_scale = 2
                        elif args.adv_iter < 6:
                            total_adv_iter = args.adv_iter
                            mul_for_scale = 1
                        else:
                            total_adv_iter = 5
                            mul_for_scale = float(total_adv_iter) / 5
                    
                    for it in range(total_adv_iter):
                        img_single.requires_grad = True
                        # print(img_single.shape)  # torch.Size([2, 3, 128, 128])
                        outputs = gcam.forward(img_single.cuda(non_blocking=True))
                        # print(outputs.shape)  # torch.Size([2, 1, 8, 8])
                        if c_idx == 0 and it == 0:
                            cam_all_classes = torch.zeros([n_classes, outputs.shape[2], outputs.shape[3]])
                        # print(list(torch.nonzero(pack['label'][0])[:, 0]), c_idx, c)
                        # print(c.requires_grad)
                        gcam.backward(ids=c)
                        regions = gcam.generate(target_layer=args.target_layer)
                        # torch.Size([2, 1, 8, 8])
                        # print(list(torch.nonzero(pack['label'][0])[:, 0]))  # [tensor(0)]
                        regions = regions[0] + regions[1].flip(-1)
                        # torch.Size([1, 8, 8])
                        ##########生产伪标签过程中可视化背景和前景################
                        # fg_cam = regions.detach().cpu().numpy()[0]
                        # fg_img = np.uint8(255 * fg_cam)
                        # pseudo_img = cv2.applyColorMap(fg_img, cv2.COLORMAP_JET)
                        # pseudo_img = cv2.resize(pseudo_img,(256,256))
                        # cv2.imwrite('fg_grad_cam'+str(it)+'.png',pseudo_img)
                        # sys.exit(0)
                        ####################################################### 
                        if it == 0:
                            init_cam = regions.detach()
                        # print(cam_all_classes.shape, regions[0].shape)
                        cam_all_classes[c_idx] += regions[0].data.cpu() * mul_for_scale
                        # print(mul_for_scale) # 1
                        logit = outputs
                        logit = F.relu(logit)
                        # print(logit.shape)  # torch.Size([2, 1, 8, 8])
                        logit = torchutils.gap2d(logit, keepdims=True)[:, :, 0, 0]
                        # print(logit.shape)  # torch.Size([2, 1])

                        valid_cat = torch.nonzero(pack['label'][0])[:, 0]
                        logit_loss = - 2 * (logit[:, c]).sum() + torch.sum(logit)
                        print(logit_loss)

                        expanded_mask = torch.zeros(regions.shape)
                        expanded_mask = add_discriminative(expanded_mask, regions, score_th=args.score_th)

                        L_AD = torch.sum((torch.abs(regions - init_cam))*expanded_mask.cuda())

                        loss = - logit_loss - L_AD * args.AD_coeff
                        print(loss)
                        model.zero_grad()
                        img_single.grad.zero_()
                        loss.backward()

                        data_grad = img_single.grad.data

                        perturbed_data = adv_climb(img_single, args.AD_stepsize, data_grad)
                        img_single = perturbed_data.detach()
                    sys.exit(0)
                outputs_cam.append(cam_all_classes)

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs_cam]), 0)
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs_cam]
            # print(highres_cam[0].shape)  # torch.Size([1, 1, 256, 256])
            sys.exit(0)
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})
            print('图片' + img_name + '.png的CAM保存成功！！！')
            # if img_name == "top_mosaic_09cm_area1_4_3":
            #     sys.exit(0)

if __name__ == '__main__':
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()
    print(n_gpus)
    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)
    # _work(0, model, dataset, args)
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)