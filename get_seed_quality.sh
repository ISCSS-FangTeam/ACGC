#!/bin/bash
python run_sample.py --train_cam_pass True  --voc12_root /home/ubt/devdata/zdy/AdvCAM/Dataset/cdy --train_list cdy12/train_aug.txt --cam_learning_rate 0.0001 --cam_batch_size 16 --cam_num_epoches 40 \
					 --cam_weights_name sess/res50_cam_cdy_time_cost.pth
# python obtain_CAM_masking.py --adv_iter 2 --AD_coeff 8 --AD_stepsize 0.08 --score_th 0.6 --voc12_root /home/ubt/devdata/zdy/AdvCAM/Dataset/cdy  --train_list cdy12/train_aug.txt --cam_weights_name sess/res50_cam_cdy.pth --cam_out_dir result/cam_adv_mask_cdy
# python run_sample.py --eval_cam_pass True --cam_out_dir result/cam_adv_mask_cdy
# python run_sample.py --cam_to_ir_label_pass True --conf_fg_thres 0.4 --conf_bg_thres 0.3 --cam_out_dir result/cam_adv_mask_cdy --voc12_root /home/ubt/devdata/zdy/AdvCAM/Dataset/cdy  --train_list cdy12/train_aug.txt --ir_label_out_dir result/ir_label_cdy 
# python run_sample.py --train_irn_pass True --irn_crop_size 256 --irn_num_epoches 20 --irn_weights_name sess/res50_irn_cdy.pth --voc12_root /home/ubt/devdata/zdy/AdvCAM/Dataset/cdy  --train_list cdy12/train_aug_building.txt --ir_label_out_dir result/ir_label_cdy --infer_list cdy12/train_aug_building.txt --irn_learning_rate 0.01
# python run_sample.py --make_sem_seg_pass True --eval_sem_seg_pass True --sem_seg_bg_thres 0.35 --cam_out_dir result/cam_adv_mask_cdy --sem_seg_out_dir result/sem_seg_cdy --irn_weights_name sess/res50_irn_cdy.pth --infer_list cdy12/train_aug_building.txt --voc12_root /home/ubt/devdata/zdy/AdvCAM/Dataset/cdy


#!/bin/bash
# python run_sample.py --train_cam_pass True  --voc12_root /home/ubt/devdata/zdy/AdvCAM/Dataset/cdy --train_list cdy12/train_aug.txt --cam_learning_rate 0.0001 --cam_weights_name sess/res50_cam_cdy.pth --cam_batch_size 16 --cam_num_epoches 40
# python obtain_CAM_masking.py --adv_iter 2 --AD_coeff 6 --AD_stepsize 0.08 --score_th 0.6 --voc12_root /home/ubt/devdata/zdy/AdvCAM/Dataset/cdy  --train_list cdy12/train_aug.txt --cam_weights_name sess/res50_cam_cdy.pth --cam_out_dir result/cam_adv_mask_cdy
# python run_sample.py --eval_cam_pass True --cam_out_dir result/cam_adv_mask_cdy
# python run_sample.py --cam_to_ir_label_pass True --conf_fg_thres 0.4 --conf_bg_thres 0.3 --cam_out_dir result/cam_adv_mask_cdy --voc12_root /home/ubt/devdata/zdy/AdvCAM/Dataset/cdy  --train_list cdy12/train_aug.txt --ir_label_out_dir result/ir_label_cdy 
# python run_sample.py --train_irn_pass True --irn_crop_size 256 --irn_num_epoches 20 --irn_weights_name sess/res50_irn_cdy.pth --voc12_root /home/ubt/devdata/zdy/AdvCAM/Dataset/cdy  --train_list cdy12/train_aug_building.txt --ir_label_out_dir result/ir_label_cdy --infer_list cdy12/train_aug_building.txt --irn_learning_rate 0.1
# python run_sample.py --make_sem_seg_pass True --eval_sem_seg_pass False --sem_seg_bg_thres 0.35 --cam_out_dir result/cam_adv_mask_cdy --sem_seg_out_dir result/sem_seg_cdy --irn_weights_name sess/res50_irn_cdy.pth --infer_list cdy12/train_aug_building.txt --voc12_root /home/ubt/devdata/zdy/AdvCAM/Dataset/cdy
