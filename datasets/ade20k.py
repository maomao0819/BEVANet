# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import os
import cv2
import numpy as np
from PIL import Image
from .base_dataset import BaseDataset

class ADE20k(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=150,
                 multi_scale=False,
                 smooth_kernel=0,
                 to_binary=True,
                 rand_scale=True,
                 flip=True,
                 ignore_label=255, 
                 base_size=520, 
                 crop_size=(512, 512),
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(ADE20k, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.smooth_kernel = smooth_kernel
        self.to_binary = to_binary

        self.rand_scale = rand_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        self.color_list = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                            [102, 255, 0], [92, 0, 255]]
        self.class_weights = None
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []

        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
            
        return files

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = os.path.join(self.root, 'ade20k', item["img"])
        label_path = os.path.join(self.root, 'ade20k', item["label"])

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )
        label = np.array(
            Image.open(label_path).convert('P')
        )
        label = self.reduce_zero_label(label)
        size = label.shape

        if 'testval' in self.list_path:
            image = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8
            )
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        if 'val' in self.list_path:
            image, label = self.resize_short_length(
                image,
                label=label,
                short_length=self.base_size,
                fit_stride=8
            )
            edge = self.gen_edge(label, edge_pad=True, edge_size=self.bd_dilate_size, multi_scale=self.multi_scale, smooth_kernel=self.smooth_kernel, to_binary=self.to_binary)
            image, label, edge = self.rand_crop(image, label, edge)
            image = self.input_transform(image, flip=True)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), edge.copy(), np.array(size), name

        image, label = self.resize_short_length(image, label, short_length=self.base_size)

        image, label, edge = self.gen_sample(image, label, 
                                             rand_scale=self.rand_scale, is_flip=self.flip, 
                                             edge_size=self.bd_dilate_size, multi_scale=self.multi_scale, 
                                             smooth_kernel=self.smooth_kernel, to_binary=self.to_binary)
        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))