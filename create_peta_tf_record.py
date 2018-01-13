
import re
import os

import tensorflow as tf
import numpy as np
import cv2


_COLOR = {'Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow'}
_HEIGHT = 224
_WIDTH = 224

FLAGS = {'data_dir': './PETA_dataset',
         'label_map_filename': "./peta_label_map_35.txt",
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


def transform_name_to_id(labels_name, map_name2id):
    """
    transform label name to id

    :param labels_name:
    :param map_name2id:
    :return: list[labels_id]
    """
    ids = []
    for l in labels_name:
        if l in map_name2id:
            ids.append(map_name2id[l])

    return ids


def load_labels_of_image(filename, map_name2id):
    """
    load image's label

    :param filename:
    :param map_name2id:
    :return: the dict of image's labels, key: string of image file index,
             value: the image's labels
    """
    num_class = len(map_name2id)

    map_img_label = {}

    for line in open(filename):
        temp = line.split()

        """image index"""
        image_index = temp[0]
        point = image_index.find('.')
        if point != -1:
            image_index = image_index[:point]

        """image labels"""
        labels_name = temp[1:]

        """transform labels to vector"""
        vector = np.zeros(num_class, dtype=np.uint8)

        labels_id = transform_name_to_id(labels_name, map_name2id)

        for l in labels_id:
            vector[l] = 1

        """add in dict"""
        map_img_label[image_index] = [line, vector]

    return map_img_label


def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (_WIDTH, _HEIGHT))
    return img


def load_images_and_labels(data_dir, name2id):
    images = []
    labels = []
    file_names = []

    dataset_name = os.path.basename(data_dir)

    data_dir = data_dir + '/archive'

    """load image's label"""
    img_labels = load_labels_of_image(data_dir + '/Label.txt', name2id)

    """load image"""
    file_list = os.listdir(data_dir)
    file_list.remove('Label.txt')

    for file in file_list:
        img = load_image(data_dir + '/' + file)

        f = file.find('_')
        if f == -1:
            index = file[:file.find('.')]
        else:
            index = file[:f]

        file_names.append(dataset_name + '/' + file)
        images.append(img)
        labels.append(img_labels[index])

    return file_names, images, labels


def create_peta_tf_record(peta_dir, name2id):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _dict_to_tf_example(file_name, image, label):
        feature = {'filename': _bytes_feature(file_name.encode('utf8')),
                   'label_str': _bytes_feature(label[0].encode('utf8')),
                   'label_vec': _bytes_feature(label[1].tostring()),
                   'image': _bytes_feature(image.tostring())}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        return example

    """open tf record file"""
    tf_filename = peta_dir + '/' + 'train.tfrecords'
    writer = tf.python_io.TFRecordWriter(tf_filename)

    dir_list = os.listdir(peta_dir)

    for subdir in dir_list:
        if os.path.isdir(peta_dir + '/' + subdir):
            print('processing ' + subdir)

            """load images and labels"""
            file_names, images, labels = load_images_and_labels(peta_dir + '/' + subdir, name2id)
            assert len(images) == len(labels)

            """write in tf record file"""
            for i in range(len(images)):
                tf_example = _dict_to_tf_example(file_names[i], images[i], labels[i])

                writer.write(tf_example.SerializeToString())

    writer.close()

    return


if __name__ == '__main__':
    """load label map, id and name"""
    # label_id2name, label_name2id = load_label_map(filename="./PETA dataset/PETA_index.txt")

    label_id2name, label_name2id = load_label_map(filename=FLAGS['label_map_filename'])

    """create tf record file"""
    create_peta_tf_record(FLAGS['data_dir'], label_name2id)

    print('create tf records success')
