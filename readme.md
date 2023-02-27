VRNN (Versatile recurrent neural network)
=============
## Description
This is the implementation of paper "Versatile recurrent 
neural network for wide types of video restoration". 
Here we provide a deblurring demo of VRNN on 
a small version of GOPRO dataset.

## System requirements
#### Prerequisites
* NVIDIA GPU + CUDA (Geforce RTX 3090 / 3090 Ti with 24GB memory, CUDA 11.1 was tested)

#### Installation
* Python 3.6+
* Pytorch 1.7.0+

## Quick start
#### Dataset
A small version of GOPRO dataset is deposited in 
```dataset/GOPRO_demo/```, which supports simple 
training and test.

#### Test and evaluation
* Run ```python test.py``` to run the trained model 
deposited in ```checkpoints/``` . 
The results will be stored in ```../results/```.
* Run ```python evaluate.py``` to obtain the 
quantitative evaluations.

#### Training
* Run ```python train.py``` to perform training 
with the default setting on the training data in 
```../data/small_dataset/train/```.

Please cite our paper in your publications if our work helps your research.

```
@article{wang2023versatile,
  title={Versatile recurrent neural network for wide types of video restoration},
  author={Wang, Yadong and Bai, Xiangzhi},
  journal={Pattern Recognition},
  volume={138},
  pages={109360},
  year={2023},
  publisher={Elsevier}
}
```
