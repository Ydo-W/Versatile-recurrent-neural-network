import cv2
import os
from para import Parameter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


if __name__ == '__main__':
    para = Parameter().args
    video_dir1 = 'results/'
    videos1 = os.listdir(video_dir1)
    video_dir2 = para.data_root + 'valid/'
    videos2 = os.listdir(video_dir2)

    PSNR, SSIM = 0.0, 0.0
    count = 0
    for video_index in range(len(videos1)):
        count = count + 1

        dirs1_name = video_dir1 + '/' +  videos1[video_index]
        dirs2_name = video_dir2 + '/' +  videos1[video_index]

        frames1 = os.listdir(dirs1_name)
        frames2 = os.listdir(dirs2_name + '/sharp/')
        frames1.sort(key=lambda x:int(x[0:4]))
        frames2.sort(key=lambda x:int(x[:-4]))

        seq_psnr, seq_ssim = 0.0, 0.0
        for frame_idx in range(len(frames1)):
            frame1 = cv2.imread(dirs1_name + '/' + frames1[frame_idx])
            frame2 = cv2.imread(dirs2_name + '/sharp/' + frames2[frame_idx + 2])
            
            frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2YCR_CB)[:, :, 0]
            frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2YCR_CB)[:, :, 0]

            frame_psnr = peak_signal_noise_ratio(frame1, frame2, data_range=255.)
            frame_ssim = structural_similarity(frame1, frame2, data_range=255.)

            seq_psnr = seq_psnr + frame_psnr / len(frames1)
            seq_ssim = seq_ssim + frame_ssim / len(frames1)

        print('Seq {:03d},  PSNR: {:.2f},  SSIM: {:.4f}'.format(video_index, seq_psnr, seq_ssim))

        PSNR = PSNR + seq_psnr / len(videos1)
        SSIM = SSIM + seq_ssim / len(videos1)

    print('PSNR: {:02f}, SSIM: {:04f}'.format(PSNR, SSIM))