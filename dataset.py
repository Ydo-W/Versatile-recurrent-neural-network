import os
import cv2
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class Crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        top, left = sample['top'], sample['left']
        new_h, new_w = self.output_size
        sample['image'] = image[top: top + new_h,
                          left: left + new_w]
        sample['label'] = label[top: top + new_h,
                          left: left + new_w]

        return sample


class Flip(object):
    def __call__(self, sample):
        flag_lr = sample['flip_lr']
        flag_ud = sample['flip_ud']
        if flag_lr == 1:
            sample['image'] = np.fliplr(sample['image'])
            sample['label'] = np.fliplr(sample['label'])
        if flag_ud == 1:
            sample['image'] = np.flipud(sample['image'])
            sample['label'] = np.flipud(sample['label'])

        return sample


class Rotate(object):
    def __call__(self, sample):
        flag = sample['rotate']
        if flag == 1:
            sample['image'] = sample['image'].transpose(1, 0, 2)
            sample['label'] = sample['label'].transpose(1, 0, 2)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[np.newaxis, :])
        label = np.ascontiguousarray(label.transpose((2, 0, 1))[np.newaxis, :])
        sample['image'] = torch.from_numpy(image).float()
        sample['label'] = torch.from_numpy(label).float()

        return sample


class deblurring_GOPRO_dataset(Dataset):
    def __init__(self, datapath, dataset_type='train', frames=12, num_ff=2, num_fb=2, crop_size=256):
        if dataset_type == 'train':
            self.transform = transforms.Compose([Crop(crop_size), Flip(), Rotate(), ToTensor()])
        else:
            self.transform = transforms.Compose([Crop(crop_size), ToTensor()])
        self.datapath = datapath + '/' + dataset_type  # './GOPRO/Train/'
        self.videos = os.listdir(self.datapath)
        self.seq_num = len(self.videos)
        self.seq_id_start = 0
        self.seq_id_end = self.seq_num - 1
        self.frames = frames
        self.num_ff = num_ff
        self.num_fb = num_fb
        self.crop_size = crop_size
        self.H = 540
        self.W = 960
        self.C = 3

    def seq_info(self, seq_idx):
        # obtain the basic info of videos
        video_dir = self.datapath + '/' + self.videos[seq_idx] + '/' + 'blur_gamma'
        files = os.listdir(video_dir)
        flag, img_num = 0, 0
        img_H, img_W, img_C = self.H, self.W, self.C
        for i in range(len(files)):
            if files[i][-4:] == '.jpg' or files[i][-4:] == '.png':
                if flag == 0:
                    flag = 1
                    img = cv2.imread(video_dir + '/' + files[i])
                    img_H, img_W, img_C = img.shape
                img_num = img_num + 1

        return img_num, img_H, img_W, img_C

    def get_index(self):
        seq_idx = random.randint(self.seq_id_start, self.seq_id_end)
        img_num, self.H, self.W, self.C = self.seq_info(seq_idx)
        frame_idx = random.randint(0, img_num - self.frames)

        return seq_idx, frame_idx

    def get_img(self, seq_idx, frame_idx, sample):
        imgs_deg = os.listdir(self.datapath + '/' + self.videos[seq_idx] + '/' + 'blur_gamma')
        imgs_deg.sort(key=lambda x:int(x[:-4]))
        img_deg_name = self.datapath + '/' + self.videos[seq_idx] + '/' + 'blur_gamma' + '/' + imgs_deg[frame_idx]
        sample['image'] = cv2.imread(img_deg_name)
        img_gt_name = self.datapath + '/' + self.videos[seq_idx] + '/' + 'sharp' + '/' + imgs_deg[frame_idx]
        sample['label'] = cv2.imread(img_gt_name)
        sample = self.transform(sample)

        return sample['image'], sample['label']

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.get_index()
        top = random.randint(0, self.H - self.crop_size)
        left = random.randint(0, self.W - self.crop_size)
        flip_lr_flag = random.randint(0, 1)
        flip_ud_flag = random.randint(0, 1)
        rotate_flag = random.randint(0, 1)
        sample = {'top': top, 'left': left,
                  'flip_lr': flip_lr_flag,
                  'flip_ud': flip_ud_flag,
                  'rotate': rotate_flag}
        imgs_deg = []
        imgs_gt = []
        for i in range(self.frames):
            img_blur, img_gt = self.get_img(seq_idx, frame_idx + i, sample)
            imgs_deg.append(img_blur)
            imgs_gt.append(img_gt)

        return torch.cat(imgs_deg, dim=0), \
               torch.cat(imgs_gt[self.num_fb:self.frames - self.num_ff], dim=0)

    def __len__(self):
        return self.seq_num * (100 - self.frames)
