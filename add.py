#!/usr/bin/python
# Author: Clara Vania

import tensorflow as tf


class AdditiveModel(object):
    """
    RNNLM using subword to word (S2W) model
    Code based on tensorflow tutorial on building a PTB LSTM model.
    https://www.tensorflow.org/versions/r0.7/tutorials/recurrent/index.html
    """
    def __init__(self, args, is_training, is_testing=False):
        self.batch_size = batch_size = args.batch_size
        self.num_steps = num_steps = args.num_steps
        self.model = model = args.model
        self.subword_vocab_size = subword_vocab_size = args.subword_vocab_size
        self.optimizer = args.optimization
        self.unit = args.unit
        self.debug = args.debug

        rnn_size = args.rnn_size
        out_vocab_size = args.out_vocab_size
        tf_device = "/gpu:" + str(args.gpu)

        if is_testing:
            self.batch_size = batch_size = 1
            self.num_steps = num_steps = 1

        if model == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif model == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        with tf.device(tf_device):
            # placeholders for data
            self._input_data = tf.placeholder(tf.float32, shape=[batch_size, num_steps, subword_vocab_size])
            if self.debug:
                print("_input_data: batch_size, num_steps, subword_vocab_size ",self._input_data)
            self._targets = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
            if self.debug:
                print("_targets: batch_size, num_steps ",self._targets)

            # ********************************************************************************
            # RNNLM
            # ********************************************************************************


            if self.debug:
                print("create rnn %s, rnn_size = %d, num_layers = %d " %(str(cell_fn), rnn_size, args.num_layers))

            lm_cell = cell_fn(rnn_size, forget_bias=0.0)
            if is_training and args.keep_prob < 1:
                lm_cell = tf.nn.rnn_cell.DropoutWrapper(lm_cell, output_keep_prob=args.keep_prob)
            # lm_cell is a column (list) of num_layers cells
            lm_cell = tf.nn.rnn_cell.MultiRNNCell([lm_cell] * args.num_layers)

            self._initial_lm_state = lm_cell.zero_state(batch_size, tf.float32)

            inputs = self._input_data
            if self.debug:
                print("inputs before split", inputs)

            if is_training and args.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, args.keep_prob)

            softmax_win = tf.get_variable("softmax_win", [subword_vocab_size, rnn_size])
            softmax_bin = tf.get_variable("softmax_bin", [rnn_size])

            # split input into a list
            inputs = tf.split(axis = 1, num_or_size_splits = num_steps, value = inputs)
            # inputs is a list of size num_steps with each tensor [batch_size, 1, sub_vocab_size]
            if self.debug:
                print("split inputs into %d results in %s" %(num_steps, str(inputs)))

            lm_inputs = []
            for input_ in inputs:
                # remove dimensions of size one, results in [batch_size, sub_vocab_size]
                input_ = tf.squeeze(input_, [1])
                input_ = tf.matmul(input_, softmax_win) + softmax_bin
                # input_ : [batch_size, rnn_size]
                lm_inputs.append(input_)
            if self.debug:
                print("list of inputs ", lm_inputs)

            lm_outputs, lm_state = tf.contrib.rnn.static_rnn(lm_cell, lm_inputs, initial_state=self._initial_lm_state)
            
            #lm_outputs is a list of num_steps tensors [batch_size, rnn_size]
            if self.debug:
                print("lm_outputs1 ",lm_outputs)
            
            lm_outputs = tf.concat(axis = 1, values = lm_outputs)
            #lm_outputs is a tensor [batch_size, num_steps * rnn_size]
            if self.debug:
                print("lm_outputs2 ",lm_outputs)
            lm_outputs = tf.reshape(lm_outputs, [-1, rnn_size])
            #lm_outputs is a tensor [batch_size * num_steps , rnn_size]
            if self.debug:
                print("lm_outputs3 ",lm_outputs)
 
            softmax_w = tf.get_variable("softmax_w", [out_vocab_size, rnn_size])
            softmax_b = tf.get_variable("softmax_b", [out_vocab_size])

            if self.debug:
                print("output vocabulary size ",out_vocab_size)

            loss_targets = tf.reshape(self._targets, [-1])
            # loss_targets dim: [batch_size * num_steps]
            if self.debug:
                print("loss targets", loss_targets)
 
            # compute cross entropy loss, logits dim: [ batch_size * num_steps, out_vocab_size ]
            logits = tf.matmul(lm_outputs, softmax_w, transpose_b=True) + softmax_b
                       
            if self.debug:
                print("logits ", logits)

            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example (
                logits = [logits],
                targets = [loss_targets],
                weights = [tf.ones([batch_size * num_steps])])

            # compute cost
            self._cost = cost = tf.reduce_sum(loss) / batch_size
            self._final_state = lm_state

            if not is_training:
                return

            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              args.grad_clip)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars))

            self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
            self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_lm_state(self):
        return self._initial_lm_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
