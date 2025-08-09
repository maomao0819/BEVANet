# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import random

from torch.nn import functional as F
from torch.utils import data

y_k_size = 6
x_k_size = 6

class BaseDataset(data.Dataset):
    def __init__(self,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image, flip=True):
        if flip:
            image = image.astype(np.float32)[:, :, ::-1]
        else:
            image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype(np.uint8)

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label, edge):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))
        edge = self.pad_image(edge, h, w, self.crop_size,
                               (0.0,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        edge = edge[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label, edge

    def multi_scale_aug(self, image, label=None, edge=None,
                        rand_scale=1, rand_crop=True):
        long_size = np.int_(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int_(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int_(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
            if edge is not None:
                edge = cv2.resize(edge, (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label, edge = self.rand_crop(image, label, edge)

        return image, label, edge

    def resize_short_length(self, image, label=None, short_length=None, fit_stride=None, return_padding=False):
        h, w = image.shape[:2]
        if h < w:
            new_h = short_length
            new_w = int(w * short_length / h + 0.5)
        else:
            new_w = short_length
            new_h = int(h * short_length / w + 0.5)
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = 0, 0
        if fit_stride is not None:
            pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
            pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=tuple(x * 255 for x in self.mean[::-1])
            )

        if label is not None:
            label = cv2.resize(
                label, (new_w, new_h),
                interpolation=cv2.INTER_NEAREST)
            if pad_h > 0 or pad_w > 0:
                label = cv2.copyMakeBorder(
                    label, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=self.ignore_label
                )
            if return_padding:
                return image, label, (pad_h, pad_w)
            else:
                return image, label
        else:
            if return_padding:
                return image, (pad_h, pad_w)
            else:
                return image

    def reduce_zero_label(self, labelmap):
        labelmap = np.array(labelmap)
        encoded_labelmap = labelmap - 1

        return encoded_labelmap

    def detect_edge(self, label, apertureSize=3, edge_pad=True, edge_size=4):
        edge = cv2.Canny(label, 0.1, 0.2, apertureSize=apertureSize)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0
        return edge

    def gen_edge(self, label, edge_pad=True, edge_size=4, multi_scale=True, smooth_kernel=0, to_binary=True):
        edge = self.detect_edge(label, apertureSize=3, edge_pad=edge_pad, edge_size=edge_size)
        if multi_scale:
            edge_5 = self.detect_edge(label, apertureSize=5, edge_pad=edge_pad, edge_size=edge_size)
            edge_7 = self.detect_edge(label, apertureSize=7, edge_pad=edge_pad, edge_size=edge_size)
            edge = (edge + edge_5 + edge_7) / 3
        if smooth_kernel > 0:
            edge = cv2.GaussianBlur(edge, (smooth_kernel, smooth_kernel), 0)
        if to_binary:
            edge = np.where(edge >= 0.5, 1.0, 0.0)
        return edge
        
    def gen_sample(self, image, label,
                   rand_scale=True, is_flip=True, edge_pad=True, edge_size=4, multi_scale=True, smooth_kernel=5, to_binary=True, flip=True):
        edge = self.gen_edge(label, edge_pad=edge_pad, edge_size=edge_size, multi_scale=multi_scale, smooth_kernel=smooth_kernel, to_binary=to_binary)

        if rand_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label, edge = self.multi_scale_aug(image, label, edge,
                                                rand_scale=rand_scale)

        image = self.input_transform(image, flip=flip)
        label = self.label_transform(label)
        

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            edge = edge[:, ::flip]

        return image, label, edge


    def inference(self, config, model, image):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]
        
        
        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )
        
        
        return pred.exp()

