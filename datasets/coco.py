# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import os
import cv2
import numpy as np
from PIL import Image
from .base_dataset import BaseDataset


class COCOStuff(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=171,
                 multi_scale=False,
                 smooth_kernel=0,
                 to_binary=True,
                 rand_scale=True,
                 flip=True,
                 ignore_label=255, 
                 base_size=640, 
                 crop_size=(640, 640),
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(COCOStuff, self).__init__(ignore_label, base_size,
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
        self.color_list = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                            [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                            [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                            [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                            [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                            [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                            [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                            [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                            [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                            [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                            [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                            [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                            [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                            [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                            [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                            [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                            [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                            [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                            [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                            [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                            [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                            [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                            [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                            [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                            [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                            [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                            [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                            [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                            [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                            [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                            [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                            [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                            [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                            [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                            [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                            [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                            [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                            [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                            [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                            [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                            [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                            [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                            [64, 192, 96], [64, 160, 64], [64, 64, 0]]
        self.class_weights = None
        self.mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
                    40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
                    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 
                    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 
                    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 
                    113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 
                    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 
                    145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 
                    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 
                    177, 178, 179, 180, 181, 182]
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

    def encode_label(self, labelmap):
        ret = np.ones_like(labelmap) * 255
        for idx, label in enumerate(self.mapping):
            ret[labelmap == label] = idx

        return ret
    
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = os.path.join(self.root, 'coco', item["img"])
        label_path = os.path.join(self.root, 'coco', item["label"])

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )
        label = np.array(
            Image.open(label_path).convert('P')
        )
        label = self.encode_label(label)
        label = self.reduce_zero_label(label)
        size = label.shape

        if 'testval' in self.list_path:
            image, border_padding = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8,
                return_padding=True
            )
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name, border_padding

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