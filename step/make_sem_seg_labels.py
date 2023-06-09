import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio

import voc12.dataloader
from misc import torchutils, indexing
from PIL import Image
import time
cudnn.enabled = True
palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]
def _work(process_id, model, dataset, args):

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()
        time_list = []
        for iter, pack in enumerate(data_loader):
            begin = time.time()
            img_name = pack['name'][0]
            # if os.path.exists(os.path.join(args.sem_seg_out_dir, img_name + '.png')):
            #     continue
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))
            # if img_name == 'top_mosaic_09cm_area1_4_3':
            #     print(edge.shape)
            #     edge1 = F.interpolate(edge.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)[0,0, :orig_img_size[0], :orig_img_size[1]]
            #     edge2 = (1-edge1.cpu().numpy().squeeze())*255
            #     print(edge2.shape)

            #     imageio.imsave(os.path.join('/home/ubt/devdata/zdy/AdvCAM', img_name + '.png'), edge2.astype(np.uint8))
            #     break
            # else:
            #     print('跳过')
            #     continue
            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '.npy', allow_pickle=True).item()

            cams = np.power(cam_dict['cam'], 1.5)
            # for cam in cams:
            #     print(cam.shape, cam.max())
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]
            ######################################
            end = time.time()
            time_list.append(end-begin)
            print(end-begin)
            continue
            ######################################
            out = Image.fromarray(rw_pred.astype(np.uint8), mode='P')
            out.putpalette(palette)
            out.save(os.path.join(os.path.join(args.sem_seg_out_dir, img_name + '_palette.png')))
            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')

        time_totao = 0.0
        for t in time_list:
            time_totao += t
        print('time', (time_totao-time_list[0]) / (len(time_list)-1))

def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    print(args.irn_weights_name)
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list,
                                                             voc12_root=args.voc12_root,
                                                             scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
