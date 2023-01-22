import torch
import torch.nn as nn
from importlib import import_module
from torch.nn.modules.loss import _Loss
from torchvision.models.vgg import vgg19
import torch.optim.lr_scheduler as lr_scheduler


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        if len(x.shape) == 5:
            b, n, c, h, w = x.shape
            x = x.reshape(b * n, c, h, w)
            y = y.reshape(b * n, c, h, w)
        perception_loss = self.loss(self.loss_network(x), self.loss_network(y))
        return perception_loss


class Optimizer:
    def __init__(self, target, lr, milestones, gamma):
        trainable = target.parameters()
        optimizer_name = 'Adam'
        module = import_module('torch.optim')
        self.optimizer = getattr(module, optimizer_name)(trainable, lr=lr)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.scheduler.step()


# Computes and stores the average value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PSNR(_Loss):
    def __init__(self):
        super(PSNR, self).__init__()
        self.rgb = 255

    def _quantize(self, x):
        return x.clamp(0, self.rgb).round()  # 压缩到0~255并临近取整

    def forward(self, x, y):
        diff = self._quantize(x) - y

        if x.dim() == 3:
            n = 1
        elif x.dim() == 4:
            n = x.size(0)
        elif x.dim() == 5:
            n = x.size(0) * x.size(1)

        mse = diff.div(self.rgb).pow(2).view(n, -1).mean(dim=-1) + 0.000001
        psnr = -10 * mse.log10()

        return psnr.mean()


def train(train_loader, model, metrics, content_loss, perception_loss, opt, epoch):
    model.train()
    print('Epoch [{:03d}], lr={:.2e}'.format(epoch, opt.get_lr()), end='||')
    loss_meter = AverageMeter()
    measure_meter = AverageMeter()

    for inputs, labels in train_loader:
        # forward
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = content_loss(outputs, labels) + 0.1 * perception_loss(outputs, labels)
        measure = metrics(outputs.detach(), labels)
        loss_meter.update(loss.detach().item(), inputs.size(0))
        measure_meter.update(measure.detach().item(), inputs.size(0))

        # backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

    print('Training: loss={:.2f}, PSNR={:.2f}'
          .format(loss_meter.avg, measure_meter.avg), end='||')
    opt.lr_schedule()


def valid(valid_loader, model, metrics):
    model.eval()
    measure_meter = AverageMeter()

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            measure = metrics(outputs.detach(), labels)
            measure_meter.update(measure.detach().item(), inputs.size(0))

    print('Valid: PSNR={:.2f}'
          .format(measure_meter.avg), end='||')
