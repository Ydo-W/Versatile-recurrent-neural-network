import argparse

class Parameter:
    def __init__(self):
        self.args = self.set_args()

    def set_args(self):
        self.parser = argparse.ArgumentParser(description='Video deblurring on GOPRO')

        # global parameters
        self.parser.add_argument('--batch_size', type=int, default=2)
        self.parser.add_argument('--resume_file', type=str, default='checkpoints/VRNN_GOPRO.pth')

        # data parameters
        self.parser.add_argument('--data_root', type=str, default='datasets/GOPRO_demo/')
        self.parser.add_argument('--frame_length', type=int, default=12)
        self.parser.add_argument('--num_ff', type=int, default=2)
        self.parser.add_argument('--num_fb', type=int, default=2)

        # training parameters
        self.parser.add_argument('--end_epoch', type=int, default=500)
        args, _ = self.parser.parse_known_args()

        return args
