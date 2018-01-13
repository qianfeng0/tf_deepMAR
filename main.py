
import re

import tensorflow as tf

import resnet

_COLOR = {'Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow'}
_NUM_CLASSES = 35
_HEIGHT = 224
_WIDTH = 224

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 1012,
    'validation': 0,
}

print(tf.__version__)

FLAGS = {'data_dir': './PETA_dataset/train.tfrecords',
         'model_dir': './model',
         'train_epochs': 50,
         'epochs_per_eval': 1,
         'batch_size': 50,
         'data_format': None,
         }


def load_label_map(filename):
    """
    load label's id and name from filename

    :param filename:
    :return: dict{name : id} and dict{id : name}
    """
    label = []
    id2name = {}
    name2id = {}
    for line in open(filename):
        while line[-1] == '\n':
            line = line[:-1]
        temp = re.split('\t', line)
        elems = [elem for elem in temp if elem != '']
        label.append(elems)

    """
    for i in range(len(label)):
        label[i][0] = i
    """

    for l in label:
        l[0] = int(l[0])
        id2name[l[0]] = l[1]
        name2id[l[1]] = l[0]

    return id2name, name2id


def read_and_decode(filename_queue):
    # serialized_ex_it = tf.python_io.tf_record_iterator(path)
    #
    # for serialized_ex in serialized_ex_it:
    #     example = tf.train.Example()
    #     example.ParseFromString(serialized_ex)
    #
    #     print(example)

    reader = tf.TFRecordReader()

    a, serialized_ex = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_ex,
                                       features={'filename': tf.FixedLenFeature([], tf.string),
                                                 'label_str': tf.FixedLenFeature([], tf.string),
                                                 'label_vec': tf.FixedLenFeature([], tf.string),
                                                 'image': tf.FixedLenFeature([], tf.string)})

    image = tf.decode_raw(features['image'], tf.uint8)
    label = tf.decode_raw(features['label_vec'], tf.uint8)
    # label_name = features['label_str']
    # file_name = features['filename']

    label = tf.reshape(label, [_NUM_CLASSES, ])
    image = tf.reshape(image, [_HEIGHT, _WIDTH, 3])

    return image, label


def preprocess_image(image, is_training):
    if is_training:
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)

    return image


def input_fn(is_training, filename, batch_size, num_epochs=1):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

    image, label = read_and_decode(filename_queue)

    image = preprocess_image(image, is_training)

    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, labels


def resnet_model_fn(features, labels, mode, params):
    network = resnet.imagenet_resnet_v2(50, _NUM_CLASSES)

    logits = network(inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {'classes': tf.round(tf.sigmoid(logits)),
                   'probabilities': tf.nn.sigmoid(logits, name='sigmoid_tensor')}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # calculate loss, which includes sigmoid cross entroy and L2 regularization
    cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

    # create a tensor name cross_entropy for logging purpose
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Scale the learning rate linearly with the batch size. When the batch size
        # is 256, the learning rate should be 0.1.
        initial_learning_rate = 0.1 * params['batch_size'] / 256
        batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
        global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 30, 60, 80, and 90 epochs.
        boundaries = [
            int(batches_per_epoch * epoch) for epoch in [30, 60, 80, 90]]
        values = [
            initial_learning_rate * decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes.
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM)

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    # accuracy
    # correct_prediction = tf.equal(tf.cast(tf.round(tf.sigmoid(logits)), tf.uint8), labels)
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    metrics = {'accuracy': accuracy}

    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def train():

    tf.logging.set_verbosity(tf.logging.INFO)

    run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=10,
                                                  log_step_count_steps=10,
                                                  save_summary_steps=10)

    resnet_classifier = tf.estimator.Estimator(
        model_fn=resnet_model_fn, model_dir=FLAGS['model_dir'], config=run_config,
        params={
            'data_format': FLAGS['data_format'],
            'batch_size': FLAGS['batch_size'],
        })

    for _ in range(FLAGS['train_epochs'] // FLAGS['epochs_per_eval']):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1)

        print('Starting a training cycle.')
        resnet_classifier.train(
            input_fn=lambda: input_fn(
                True, FLAGS['data_dir'], FLAGS['batch_size'], FLAGS['epochs_per_eval']),
            hooks=[logging_hook])

        print('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(
            input_fn=lambda: input_fn(False, FLAGS['data_dir'], FLAGS['batch_size']))
        print(eval_results)


def main():
    """load label map, id and name"""
    # label_id2name, label_name2id = load_label_map(filename="./PETA dataset/PETA_index.txt")

    # images, labels = input_fn(True, './PETA dataset/train.tfrecords', 100, 2)
    #
    # test = resnet_model_fn(images, labels, tf.estimator.ModeKeys.TRAIN)


    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        # sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train()

        # for i in range(1000):
        #     a = sess.run([images, labels])
        #     # a = sess.run([logits, probabilities, cross_entropy])

        coord.request_stop()
        coord.join(threads)

    # train()




if __name__ == '__main__':
    print(10)
    main()
