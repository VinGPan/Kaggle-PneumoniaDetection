import tensorflow as tf
from keras.backend.common import epsilon
import keras.losses
from keras.models import *


class Loss:
    def __init__(self, typ):
        self.loss = None
        if typ == "categor_iou":
            self.loss = Loss.categor_iou
        else:
            assert 0
        pass

    @staticmethod
    def iou(target, output):
        output /= tf.reduce_sum(output, len(output.get_shape()) - 1, True)
        intersection = tf.reduce_sum(target * output, 1)
        intersection = intersection * tf.constant([0.0, 1.0])
        intersection = tf.reduce_sum(intersection, 1)

        den1 = tf.reduce_sum(target, 1) * tf.constant([0.0, 1.0])
        den2 = tf.reduce_sum(output, 1) * tf.constant([0.0, 1.0])
        den1 = tf.reduce_sum(den1, 1)
        den2 = tf.reduce_sum(den2, 1)
        score1 = intersection / (den1 + den2 - intersection + epsilon())

        back_grd = 1.0 - K.clip(den1, 0, 1)
        score2 = tf.reduce_mean(output, 1) * tf.constant([1.0, 0.0])
        score2 = tf.reduce_sum(score2, 1)
        score2 = score2 * back_grd

        return score1 + score2

    @staticmethod
    def iou_loss(target, output):
        return 1. - Loss.iou(target, output)

    @staticmethod
    def categor_iou(target, output):
        l1 = keras.losses.categorical_crossentropy(target, output)
        l1 = tf.reduce_mean(l1, 1) * .5
        return l1 + Loss.iou_loss(target, output) * 0.5

