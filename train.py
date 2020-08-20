import argparse
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

import cifar10
from model import model_fn


tf.get_logger().propagate = False


def is_chief():
    return hvd.rank() == 0


def data_chunk(x_train, y_train):
    x_train_chunk = x_train[hvd.rank()::hvd.size()]
    y_train_chunk = y_train[hvd.rank()::hvd.size()]
    return x_train_chunk, y_train_chunk


def preprocess(images):
    return 2 * (images / 255. - 0.5)


def main(_):
    hvd.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    args = parser.parse_args()

    # Load all CIFAR10
    (x_train, y_train), (x_test, y_test) = cifar10.data()

    # Select train data for worker
    x_train, y_train = data_chunk(x_train, y_train)

    # Preprocess train and test images
    x_train, x_test = preprocess(x_train), preprocess(x_test)

    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir='checkpoints',
                                       params={'lr': args.lr})

    train_fn = tf.estimator.inputs.numpy_input_fn(x={"images": x_train},
                                                  y=np.squeeze(y_train),
                                                  batch_size=args.batch_size,
                                                  num_epochs=1,
                                                  shuffle=True)
    
    eval_fn = tf.estimator.inputs.numpy_input_fn(x={"images": x_test},
                                                 y=np.squeeze(y_test),
                                                 batch_size=1,
                                                 num_epochs=1,
                                                 shuffle=False)

    for _ in range(args.num_epochs):
        estimator.train(input_fn=train_fn, hooks=hooks)

    if is_chief():
        estimator.evaluate(input_fn=eval_fn)


if __name__ == "__main__":
    tf.app.run()

