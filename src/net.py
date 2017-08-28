import math
import numpy as np
import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import initializers
from chainer import function
from chainer import link
from chainer.utils import array
from chainer.utils import type_check

class BatchConv2D(chainer.Chain):
    def __init__(self, ch_in, ch_out, ksize, stride=1, pad=0, activation=F.relu):
        super(BatchConv2D, self).__init__(
            conv=L.Convolution2D(ch_in, ch_out, ksize, stride, pad),
            bn=L.BatchNormalization(ch_out),
        )
        self.activation=activation

    def __call__(self, x):
        h = self.bn(self.conv(x))
        if self.activation is None:
            return h
        return self.activation(h)

class VGG(chainer.Chain):
    def __init__(self, base=64):
        super(VGG, self).__init__()
        with self.init_scope():
            self.bconv1_1 = BatchConv2D(1, base, 3, stride=1, pad=1)
            self.bconv1_2 = BatchConv2D(base, base, 3, stride=1, pad=1)
            self.bconv2_1 = BatchConv2D(base, base * 2, 3, stride=1, pad=1)
            self.bconv2_2 = BatchConv2D(base * 2, base * 2, 3, stride=1, pad=1)
            self.bconv3_1 = BatchConv2D(base * 2, base * 4, 3, stride=1, pad=1)
            self.bconv3_2 = BatchConv2D(base * 4, base * 4, 3, stride=1, pad=1)
            self.bconv4_1 = BatchConv2D(base * 4, base * 4, 3, stride=1, pad=1)
            self.bconv4_2 = BatchConv2D(base * 4, base * 4, 3, stride=1, pad=1)
            self.fc = L.Linear(base * 4, 10)

    def __call__(self, x):
        h = self.bconv1_1(x)
        h = self.bconv1_2(h)
        h = F.max_pooling_2d(h, 2)
        h = self.bconv2_1(h)
        h = self.bconv2_2(h)
        h = F.max_pooling_2d(h, 2)
        h = self.bconv3_1(h)
        h = self.bconv3_2(h)
        h = self.bconv4_1(h)
        h = self.bconv4_2(h)
        h = F.average_pooling_2d(h, 8)
        h = self.fc(h)
        return h

VGG_LAYERS = [
    'bconv1_1',
    'bconv1_2',
    'bconv2_1',
    'bconv2_2',
    'bconv3_1',
    'bconv3_2',
    'bconv4_1',
    'bconv4_2',
    'fc',
]
