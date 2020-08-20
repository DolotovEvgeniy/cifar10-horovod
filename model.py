import horovod.tensorflow as hvd
import tensorflow as tf


def model_fn(features, labels, mode, params):
    inputs = features['images']
    conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    flatten = tf.reshape(pool2, [-1, 2048])
    dense = tf.layers.dense(inputs=flatten, units=128, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=10)

    predictions = tf.argmax(input=logits,axis=1)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=params['lr'] * hvd.size())
        opt = hvd.DistributedOptimizer(opt)
        train_op = opt.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    else:
        accuracy = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=accuracy)

