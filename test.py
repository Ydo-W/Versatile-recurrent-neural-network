import os
import cv2
import VRNN
import time
import torch
import numpy as np
from os.path import join
from para import Parameter


if __name__ == '__main__':
    # Load the model
    para = Parameter().args
    model = VRNN.Model(2, 2, 18).cuda()
    checkpoint = torch.load(para.resume_file)
    model.load_state_dict(checkpoint['model'])
    print('Model successfully loaded...')
    
    # Test_path
    blur_path = para.data_root + 'valid/'
    result_path = 'results/'
    os.makedirs(result_path, exist_ok=True)
    
    # Seqs
    blur_videos = os.listdir(blur_path)
    blur_videos.sort()

    # Start deblurring
    for video_idx in range(len(blur_videos)):
        print('{:03d} seq being processed...'.format(video_idx))

        img_dir = blur_path + blur_videos[video_idx] + '/blur_gamma'
        imgs = os.listdir(img_dir)
        imgs.sort(key=lambda x:int(x[:-4]))
        out_dir = result_path + blur_videos[video_idx]
        os.makedirs(out_dir, exist_ok=True)

        input_seq = []
        for image_index in range(len(imgs)):
            frame_name = img_dir + '/' +  imgs[image_index]
            image = cv2.imread(frame_name)
            image = np.ascontiguousarray(image.transpose((2, 0, 1))[np.newaxis, :])
            input_seq.append(image)
        input_seq = np.concatenate(input_seq)[np.newaxis, :]
        input_seq = torch.from_numpy(input_seq).float().cuda()

        start = 0
        test_frames = 12
        end = test_frames
        while (True):
            input = input_seq[:, start:end, :, :, :]
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                output_seq = model(input).clamp(0, 255).squeeze()
            for frame_idx in range(end - start - 4):
                img_deblur = output_seq[frame_idx]
                img_deblur = img_deblur.detach().cpu().numpy().transpose((1, 2, 0))
                cv2.imwrite(join(out_dir, '{:04d}_deblur.png'.format(frame_idx + start)), img_deblur)
            if end == len(imgs):
                break
            else:
                start = end - 2 - 2
                end = start + test_frames
                if end > len(imgs):
                    end = len(imgs)
                    start = end - test_frames

