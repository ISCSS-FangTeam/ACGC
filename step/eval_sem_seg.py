import cv2
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt # matplotlib.use('agg')必须在本句执行前运行
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio

def run(args):
    # dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
    # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    labels = []
    ids = []
    n_img = 0
    root = '/home/ubt/devdata/zdy/AdvCAM/Dataset/cdy/SegmentationClassAug'
    with open('/home/ubt/devdata/zdy/AdvCAM/cdy12/train_aug_building.txt', 'r') as f:
        file = f.readlines()
        for i in range(0, len(file)):
            file[i] = file[i].rstrip('\n')
            img = cv2.imread(os.path.join(root,file[i]+'.png'),cv2.IMREAD_GRAYSCALE)  # 洪旺的数据集只能用最后一个通道，DLRSD数据集可以用任意一个通道
            ids.append(file[i])
            # img是0和22
            new_img = img.copy()
            new_img = new_img / 255
            new_img = new_img.astype(int)
            # print(new_img.dtype)
            # new_img.dtype = 'int64' 这样做会报错
            labels.append(new_img) 
    for i, id in enumerate(ids):
        cls_labels = imageio.imread(os.path.join(args.sem_seg_out_dir, id + '.png')).astype(np.uint8)
        # cls_labels = imageio.imread(os.path.join("/home/ubt/devdata/zdy/AdvCAM/result/ir_label_cdy", id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())
        n_img += 1

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:2, :2]
    print(confusion)
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator
    print("total images", n_img)
    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    precision = gtjresj / (fp * denominator + gtjresj)
    recall = gtjresj / (fn * denominator + gtjresj)
    F_score = 2 * (precision * recall) / (precision + recall)
    print({'precision': precision, 'recall': recall, 'F_score': F_score})
    print({'iou': iou, 'miou': np.nanmean(iou)})
    plot(confusion)

def plot(matrix):
        # matrix = self.matrix

        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]     # 归一化
        matrix = np.around(matrix, decimals=2)

        num_classes = 2
        labels = ['background','building']
        # print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(num_classes), labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(num_classes), labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(num_classes):
            for y in range(num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = matrix[y, x]
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
        plt.savefig("./pseudo.png")