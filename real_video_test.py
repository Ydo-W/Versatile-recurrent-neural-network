import torch
import numpy as np
import os
import BTSRNN_pro as BTSRNN
import cv2
from para import Parameter
import datetime
import time

if __name__ == '__main__':
    para = Parameter().args
    model = BTSRNN.Model(2, 2, 18).cuda()
    checkpoint = torch.load(para.resume_file)
    model.load_state_dict(checkpoint['model'])
    print('Model successfully loaded...')
    
    # test_path
    test_path = 'test_videos/'
    # turbu_path = test_path + 'test/'
    turbu_path = 'test_videos/'
    # result_path = test_path + para.resume_file[11:-4] + '/'
    result_path = turbu_path + para.resume_file[11:-4] + '/'
    os.makedirs(result_path, exist_ok=True)
    
    # videos
    turbu_videos = os.listdir(turbu_path)

    pre = datetime.datetime.now()
    count = 0
    for video_idx in range(len(turbu_videos)):
        print('{:03d} videos being processed...'.format(video_idx))

        input_video_path = turbu_path + turbu_videos[video_idx]
        print(input_video_path)
        out_video_path = result_path + 'result.avi'
        input_video = cv2.VideoCapture(input_video_path)
        frame_w, frame_h = int(input_video.get(3)), int(input_video.get(4))  # 原始视频宽、高
        adjust_h, adjust_w = int(frame_h/4)*4, int(frame_w/4)*4  # 将宽高调整为4的倍数，便于处理
        fps, frames_num = input_video.get(5), input_video.get(7)
        out_video = cv2.VideoWriter(out_video_path, fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                    fps=fps, frameSize=(frame_w, frame_h), isColor=True)

        # --------- load input data ------------------------------------------
        input_seq = []
        for image_index in range(int(frames_num)):
            rval, frame_input = input_video.read()
            frame_input = cv2.resize(frame_input, dsize = (adjust_w, adjust_h))
            frame_input = np.ascontiguousarray(frame_input.transpose((2, 0, 1))[np.newaxis, :])  # (1, 3, h, w)
            input_seq.append(frame_input)
        input_seq = np.concatenate(input_seq)[np.newaxis, :]  # (1, 15, 3, h, w)

        # --------- restoration ----------------------------------------------
        # start
        test_frames = 12
        start, end = 0, test_frames
        process_number = 0
        while True:
            print(start, end)
            torch.cuda.empty_cache()
            theInput = input_seq[:, start:end, :, :, :]
            model.eval()
            with torch.no_grad():
                theInput = torch.from_numpy(theInput).float().cuda()
                start_time = time.time()
                restoration = model(theInput)                           # (1, f-4, 1, h, w)
                end_time = time.time()
                print('time per frame:{:.4f}s'.format((end_time - start_time) / 11))
                output_seq = restoration.clamp(0, 255).squeeze()            # (f-4, 3, h, w)

            for frame_idx in range(test_frames - 2*para.num_ff):
                if process_number < start+frame_idx+1:
                    img_deTurbulence = output_seq[frame_idx]
                    img_deTurbulence = img_deTurbulence.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
                    img_deTurbulence = cv2.resize(img_deTurbulence, dsize = (frame_w, frame_h))
                    out_video.write(img_deTurbulence)
                    process_number += 1
            if end == int(frames_num):
                break
            else:
                start = end - 2*para.num_ff
                end = start + test_frames
                if end > int(frames_num):
                    end = int(frames_num)
                    start = end - test_frames
        
        out_video.release()
        input_video.release()