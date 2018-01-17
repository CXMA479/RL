import mxnet as mx
import numpy as np


net=mx.gluon.nn.Sequential()
net.add(mx.gluon.nn.Dense(3))
x=mx.nd.random.uniform(0,1,(1,10))
outgrad = mx.nd.array([ [0, -1, 0] ]) # label as negative if wanna score this bit
net.initialize()
trainer = mx.gluon.Trainer(net.collect_params(), 'sgd')

for i in xrange(4):
    with mx.autograd.record():
        y=net(x)
        y=mx.nd.softmax(y)
        print y
        y.backward(outgrad)
    trainer.step(1)

