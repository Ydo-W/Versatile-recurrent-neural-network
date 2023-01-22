import os
import time
import VRNN
import utils
import torch
import random
import dataset
import numpy as np
import torch.nn as nn
from para import Parameter
from torch.utils.data import DataLoader


if __name__ == '__main__':
    para = Parameter().args
    batch_size = para.batch_size

    # Set the random seed
    torch.manual_seed(39)
    torch.cuda.manual_seed(39)
    random.seed(39)
    np.random.seed(39)

    # Model
    model = VRNN.Model(2, 2, 18).cuda()
    model_parameters = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model params: {:4f}M'.format(model_parameters / 1000 / 1000))
    optimizer = utils.Optimizer(model, 1e-4, milestones=[200, 400], gamma=0.5)

    # Metrics and Losses
    metrics = utils.PSNR().cuda()
    content_loss = nn.L1Loss().cuda()
    perception_loss = utils.PerceptualLoss().cuda()  # Vgg19

    # Dataset
    path = para.data_root
    ff, fb = para.num_ff, para.num_fb
    train_dataset = dataset.deblurring_GOPRO_dataset(path, 'train', para.frame_length, ff, fb, 256)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_train: ', len(train_dataset))

    valid_dataset = dataset.deblurring_GOPRO_dataset(path, 'valid', para.frame_length, ff, fb, 256)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_valid: ', len(valid_dataset))

    # Start training
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    num_epochs = para.end_epoch
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        utils.train(train_loader, model, metrics,
                    content_loss, perception_loss, optimizer, epoch)
        utils.valid(valid_loader, model, metrics)
        end = time.time()
        print('Time:{:.2f}s'.format(end - start))
        if epoch % 20 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'opt': optimizer.optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, './checkpoints/VRNN_GOPRO_%s.pth' % (str(epoch)))

