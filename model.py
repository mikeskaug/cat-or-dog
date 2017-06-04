import tensorflow as tf

# Here we define a deep convolutional neural network model
#
# All of the tf.summary statements add things to the global summary which can be visualized in TensorBoard
# to run TensorBoard, run this command: python -m tensorflow.tensorboard --logdir='./output'
# and then browse to localhost:6006


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def variable_summaries(var):
    # Attach summaries to variables for inspection in TensorBoard
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# A layer that combines convolution, ReLU activation and 2x2 maxpooling
def conv_max_layer(x, kern_x, kern_y, channels, features):
    W = weight_variable([kern_x, kern_y, channels, features])
    variable_summaries(W)
    b = bias_variable([features])
    variable_summaries(b)
    conv = tf.nn.relu(conv2d(x, W) + b)
    out_tensor = max_pool_2x2(conv)
    return out_tensor


def fully_connected(layer, dims, features, units):
    W = weight_variable([dims * dims * features, units])
    b = bias_variable([units])
    layer_flat = tf.reshape(layer, [-1, dims * dims * features])
    fc = tf.nn.relu(tf.matmul(layer_flat, W) + b)
    return fc


# Assemble multiple convolution/maxpool layers, two fully connected layers,
# two dropout layers and a readout layer
def cnn_model(images, mode):
    x = tf.reshape(images, [-1, 64, 64, 3])
    # normalize the input images to have mean=0 and unit stddev
    norm_images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x)
    tf.summary.image('input_images', norm_images, max_outputs=6)

    with tf.name_scope('layer-1'):
        layer1 = conv_max_layer(norm_images, 3, 3, 3, 32)  # 64 -> 32
        tf.summary.image('layer1_maps', tf.slice(layer1, [0, 0, 0, 0], [-1, -1, -1, 1]), max_outputs=6)
        tf.summary.histogram('layer1', layer1)

    with tf.name_scope('layer-2'):
        layer2 = conv_max_layer(layer1, 3, 3, 32, 64)  # 32 -> 16
        tf.summary.image('layer2_maps', tf.slice(layer2, [0, 0, 0, 0], [-1, -1, -1, 1]), max_outputs=6)
        tf.summary.histogram('layer2', layer2)

    with tf.name_scope('layer-3'):
        layer3 = conv_max_layer(layer2, 3, 3, 64, 128)  # 16 -> 8
        tf.summary.image('layer3_maps', tf.slice(layer3, [0, 0, 0, 0], [-1, -1, -1, 1]), max_outputs=6)
        tf.summary.histogram('layer3', layer3)

    layer4 = fully_connected(layer3, 8, 128, 512)
    layer5 = tf.layers.dropout(inputs=layer4, rate=0.4, training=mode == 'training')
    layer6 = tf.layers.dense(inputs=layer5, units=128, activation=tf.nn.relu)
    layer7 = tf.layers.dropout(inputs=layer6, rate=0.4, training=mode == 'training')
    logits = tf.layers.dense(inputs=layer7, units=2)

    return logits


def loss(logits, labels):
    tf.summary.histogram('logits', tf.nn.softmax(logits))
    labels = tf.to_int64(labels)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.reduce_mean(cross_entropy)


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluate(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
