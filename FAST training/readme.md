**Requirements:**
Our code has been tested under Python 3.6 and Python 3.7. [Pytorch](https://pytorch.org/get-started/locally/), and [apex](https://github.com/NVIDIA/apex) are necessary for FAST training setting. 

```shell
# Install pytorch
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install apex
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ python setup.py install --cpp_ext --cuda_ext # This works for us.
```

The publicly available training code of PGD, U-FGSM, and FREE can be found at [here](https://github.com/locuslab/fast_adversarial/tree/master/CIFAR10).