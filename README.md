Fashion MNIST classifier using Chainer
====

Fashion MNIST (https://github.com/zalandoresearch/fashion-mnist) classifier.

# Requirements

* [Chainer](https://chainer.org/) 2.0
* [Cupy 1.0.0](http://docs.cupy.chainer.org/en/stable/)
* [matplotlib](http://matplotlib.org/)

# Usage

```
$ python src/train.py -g 0
```

Options:
* `-g (--gpu) <int>`: Optional  
GPU device ID. Negative value indicates CPU (default: -1)
* `-m (--model) <model file path>`: Optional  
Model file path to load (default: None)
* `-b (--batch_size) <int>`: Optional  
Mini batch size (default: 128)
* `-p (--prefix) <str>`: Optional  
Prefix of saved model file. (default: `fashion_<base size>`)
* `--epoch <int>`: Optional  
Training epochs (default: 300)
* `--save-epoch <int>`: Optional  
Epoch interval to save model parameter file. 0 indicates model parameter is not saved at fixed intervals. Note that the best accuracy model is always saved even if this parameter is 0. (default: 0)
* `--optimizer <str>`: Optional  
Optimizer name (`sgd` or `adam`, default: sgd)
* `--lr <float>`: Optional  
Initial learning rate for SGD (default: 0.1)
* `--alpha <float>`: Optional  
Initial alpha for Adam (default: 0.001)
* `--lr-decay-iter <int>`: Optional  
Iteration interval to decay learning rate. Learning rate is decay to 1/10 at this intervals. (default: 100)
* `--weight-decay <float>`: Optional  
Weight decay (default: 0.0001)
* `--seed <int>`: Optional  
Random seed (default: 1)

# Algorithm explanation

* VGG like neural network model
* Data augmentation
    * random crops
    * random horizontal flips
    * random erasing (see https://arxiv.org/abs/1708.04896)

# License

MIT license
