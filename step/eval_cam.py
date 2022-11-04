import numpy as np
import os
import six
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import cv2
from PIL import Image
import imageio

def run(args):
    # dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
                     64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
                     0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
                     64,64,0,  192,64,0,  64,192,0, 192,192,0]

    preds = []
    labels = []
    ids = []
    # ---------------------------------------------------------------------------------------
    root = '/home/ubt/devdata/zdy/AdvCAM/Dataset/cdy/SegmentationClassAug'
    with open('/home/ubt/devdata/zdy/AdvCAM/cdy12/train_aug.txt', 'r') as f:
        file = f.readlines()
        for i in range(0, len(file)):
            file[i] = file[i].rstrip('\n')
            img = cv2.imread(os.path.join(root,file[i]+'.png'),cv2.IMREAD_GRAYSCALE)  # 洪旺的数据集只能用最后一个通道，DLRSD数据集可以用任意一个通道
            if (img == 0).all():
            	# print('全为背景！！')
            	continue
            ids.append(file[i])
            # img是0和22
            new_img = img.copy()
            new_img = new_img / 255
            new_img = new_img.astype(int)
            # print(new_img.dtype)
            # new_img.dtype = 'int64' 这样做会报错
            labels.append(new_img)  
        
    # ---------------------------------------------------------------------------------------

    n_images = 0
    for i, id in enumerate(ids):
        n_images += 1
        # print(os.path.join(args.cam_out_dir, id + '.npy'))
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']  # high_res指的是high_resolution
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        # print(cams.shape) # (2, 256, 256) print(keys.shape) 0 1
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        new_cls_labels = cls_labels.copy()
        # print(np.unique(new_cls_labels))
        # where_0 = np.where(new_cls_labels == 0)
        # where_1 = np.where(new_cls_labels == 1)
        # where_2 = np.where(new_cls_labels == 2)
        # # new_cls_labels[where_1] = 0
        # new_cls_labels[where_2] = 1
        # print(np.unique(new_cls_labels))
        # print(np.max(new_cls_labels))
        preds.append(new_cls_labels)
        # labels.append(dataset.get_example_by_keys(i, (1,))[0])
        # 测试消融实验
        # out = Image.fromarray(new_cls_labels.astype(np.uint8), mode='P')
        # out.putpalette(palette)
        # out.save(os.path.join(os.path.join("/home/ubt/devdata/zdy/AdvCAM/result/Adv+CAM", id + '_palette.png')))
        # imageio.imsave(os.path.join("/home/ubt/devdata/zdy/AdvCAM/result/Adv+CAM", id + '.png'), new_cls_labels.astype(np.uint8)) 
        # 测试消融实验
    
    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)  # 真实值
    resj = confusion.sum(axis=0)  # 预测的
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    # fp = 1. - gtj / denominator
    # fn = 1. - resj / denominator
    iou = gtjresj / denominator
    # ---------------------
    # print(fp[0], fn[0])
    # print(np.mean(fp[1:]), np.mean(fn[1:]))

    # precision = gtjresj / (fp * denominator + gtjresj)
    # recall = gtjresj / (fn * denominator + gtjresj)
    # F_score = 2 * (precision * recall) / (precision + recall)
    # print({'precision': precision, 'recall': recall, 'F_score': F_score})
    # print({'iou': iou, 'miou': np.nanmean(iou)})
    # sys.exit(0)
    # ---------------------
    
    print("threshold:", args.cam_eval_thres, 'iou:', iou, 'miou:', np.nanmean(iou), "i_imgs", n_images)
    # among_predfg_bg表示背景像素被预测为建筑物类的比例，越小越好
    print('among_predfg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))

    return np.nanmean(iou)

def plot(matrix):
        # matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
# def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
#     """Collect a confusion matrix.
#     The number of classes :math:`n\_class` is
#     :math:`max(pred\_labels, gt\_labels) + 1`, which is
#     the maximum class id of the inputs added by one.
#     Args:
#         pred_labels (iterable of numpy.ndarray): See the table in
#             :func:`chainercv.evaluations.eval_semantic_segmentation`.
#         gt_labels (iterable of numpy.ndarray): See the table in
#             :func:`chainercv.evaluations.eval_semantic_segmentation`.
#     Returns:
#         numpy.ndarray:
#         A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
#         The :math:`(i, j)` th element corresponds to the number of pixels
#         that are labeled as class :math:`i` by the ground truth and
#         class :math:`j` by the prediction.
#     """
#     pred_labels = iter(pred_labels)
#     gt_labels = iter(gt_labels)

#     n_class = 0
#     confusion = np.zeros((n_class, n_class), dtype=np.int64)
#     for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
#         if pred_label.ndim != 2 or gt_label.ndim != 2:
#             raise ValueError('ndim of labels should be two.')
#         if pred_label.shape != gt_label.shape:
#             raise ValueError('Shape of ground truth and prediction should'
#                              ' be same.')
#         pred_label = pred_label.flatten()
#         gt_label = gt_label.flatten()
        
#         # Dynamically expand the confusion matrix if necessary.
#         lb_max = np.max((pred_label, gt_label))
        
#         if lb_max >= n_class:
#             expanded_confusion = np.zeros((lb_max + 1, lb_max + 1), dtype=np.int64)
#             # sys.exit(0)
#             expanded_confusion[0:n_class, 0:n_class] = confusion

#             n_class = lb_max + 1
#             confusion = expanded_confusion

#         # Count statistics from valid pixels.
#         mask = gt_label >= 0
#         confusion += np.bincount(
#             n_class * gt_label[mask].astype(int) +
#             pred_label[mask], minlength=n_class**2).reshape((n_class, n_class))

#     for iter_ in (pred_labels, gt_labels):
#         # This code assumes any iterator does not contain None as its items.
#         if next(iter_, None) is not None:
#             raise ValueError('Length of input iterables need to be same')
#     return confusion