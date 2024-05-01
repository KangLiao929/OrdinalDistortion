# A Deep Ordinal Distortion Estimation Approach for Distortion Rectification
## Introduction
This is the official implementation for [OrdinalDistortion](https://arxiv.org/abs/2007.10689) (IEEE TIP'21).

[Kang Liao](https://kangliao929.github.io/), [Chunyu Lin](http://faculty.bjtu.edu.cn/8549/), [Yao Zhao](http://mepro.bjtu.edu.cn/zhaoyao/e_index.htm)
![](https://github.com/KangLiao929/OrdinalDistortion/blob/main/assets/ordinal_distortion.png) 
> ### Problem
> Given a radial distortion image capture by wide-angle lens, this work aims to predict the ordinal distortion and rectify the distortion.
>  ### Features
>  * Propose a learning-friendly representatiion (ordinal distortion) for wide-angle image rectification, which is customized for neural networks and solves a more straightforward estimation problem than the traditional distortion parameter regression
>  * Ordinal distortion is homogeneous as all its elements share a similar magnitude and description, compared with the heterogeneous distortion parameter
>  * Ordinal distortion can be estimated using only a part of a wide-angle image, enabling higher efficiency of rectification algorithms

![](https://github.com/KangLiao929/OrdinalDistortion/blob/main/assets/method_comparison.png) 

## Requirements
- Python 3.5.6 (or higher)
- Tensorflow 1.12.0
- Keras 2.2.4
- OpenCV
- numpy
- matplotlib
- scikit-image

## Installation

```bash
git clone https://github.com/KangLiao929/OrdinalDistortion.git
cd OrdinalDistortion/
```

## Getting Started & Testing

- Download the pretrained models through the following links ([GoogleDrive](https://drive.google.com/file/d/1E9-rvypayfCrYJCZL5qkIbCY-whCvzI7/view?usp=sharing)), and put them into `weights/`. 
- To test images in a folder, you can call `test.py` with the opinion `--test_path`. For example:

  ```bash
  python test.py --test_num 50 --test_path "./imgs/*.jpg" --save_weights_path './weights/OrdinalDistortionNet.h5' --save_img_path "./results/"
  ```
  or write / modify `test.sh` according to your own needs, then execute this script as (Linux platform):  
  ```bash
  sh ./test.sh
  ```
The visual evaluations will be saved in the folder `./results/`.

## Training
- Generate the training dataset into the path `dataset/train/`.
- To train OrdinalDistortionNet, you can call `train.py` with the opinion `--train_path`. For example:
  ```shell
  python train.py --train_num 20000 --train_path "./dataset/train/A/*.jpg" --save_weights_path "./weights/" 
  ```
  or write / modify `train.sh` according to your own needs, then execute this script as:  
  ```bash
  sh ./train.sh
  ```

## Citation

If our solution is useful for your research, please consider citing:

    @article{liao2021deep,
      title={A Deep Ordinal Distortion Estimation Approach for Distortion Rectification},
      author={Liao, Kang and Lin, Chunyu and Zhao, Yao},
      journal={IEEE Transactions on Image Processing},
      volume={30},
      pages={3362--3375},
      year={2021}
    }