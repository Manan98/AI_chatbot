# -*- coding: utf-8 -*-

# Building the Seq2Seq Model
import tensorflow as tf


# Creating placeholders for the inputs and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning-rate')
    keep_prob = tf.placeholder(tf.foat32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

# Creating the Encoder RNN Layer

