import numpy as np
import tensorflow as tf
import random
import datetime

timesteps = 180
num_input_cn = 32
max_seq_len_cn = num_input_cn * timesteps
num_input_6dof = 12
max_seq_len_6dof = num_input_6dof * timesteps

def loadData(filePath, max_seq_len):
    data_out = []
    data_len = []
    data = open(filePath)
    data_arr = data.readlines()
    data_rows = len(data_arr)
    print('bags of dataset is %d' %data_rows)

    for i in range(data_rows):
        data_arr[i] = data_arr[i].strip()
        data_arr[i] = data_arr[i].split(" ")
        data_i = np.array(data_arr[i])
        data_i = data_i.astype(float)
        data_i_len = data_i.shape[0]
        data_len.append(data_i_len)
        for j in range(max_seq_len - data_i_len):
            data_i = np.append(data_i, 0)
        data_out = np.append(data_out, data_i)
    data_out = np.reshape(data_out, (data_rows, max_seq_len))
    data.close()
    return data_out, data_len

def gen_next_batch_cn(X_train, y_train, train_len, batch_size, indices, step):
    X_train_out = X_train[indices[step*batch_size:batch_size*(step+1)]]
    y_train_out = y_train[indices[step*batch_size:batch_size*(step+1)]]
    train_len_tmp = np.array(train_len)
    train_len_out = train_len_tmp[indices[step*batch_size:batch_size*(step+1)]]
    batchSizeO = min(X_train.shape[0]-step*batch_size, batch_size)
    return X_train_out, y_train_out, train_len_out, batchSizeO

def gen_next_batch_6dof(X_train, y_train, train_len, batch_size, indices, step):
    X_train_out = X_train[indices[step*batch_size:batch_size*(step+1)]]
    y_train_out = y_train[indices[step*batch_size:batch_size*(step+1)]]
    train_len_tmp = np.array(train_len)
    train_len_out = train_len_tmp[indices[step*batch_size:batch_size*(step+1)]]
    return X_train_out, y_train_out, train_len_out

# network parameter
learning_rate = 0.00001
epoch = 500
batch_size = 64
times = 10

keep_prob = 0.5
num_layers = 2
n_hidden = 256  # hidden layer num of features
n_classes = 1

weights = {'cn': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
            '6dof': tf.Variable(tf.random_normal([n_hidden, n_hidden])),
            'out': tf.Variable(tf.random_normal([n_hidden*2, n_classes]))}
biases = {'cn': tf.Variable(tf.random_normal([n_hidden])),
            '6dof': tf.Variable(tf.random_normal([n_hidden])),
            'out': tf.Variable(tf.random_normal([n_classes]))}

drop_keep_rate = tf.placeholder(tf.float32, name="dropout_keep")
seqlen = tf.placeholder(tf.int32, [None])
x_cn = tf.placeholder("float", [None, timesteps, num_input_cn])
x_6dof = tf.placeholder("float", [None, timesteps, num_input_6dof])
y = tf.placeholder("float", [None, n_classes])

def dynamicRNN_stack(x_cn, x_6dof, seqlen, weights, biases, keep_prob, num_layers):
    # dynamicRNN_cn
    x_cn = tf.unstack(x_cn, timesteps, 1)
    with tf.variable_scope('cn'):
        # cells_fw = []
        # cells_bw = []
        # for _ in range(num_layers):
        #     cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = keep_prob)
        #     cells_fw.append(cell)
        #     cells_bw.append(cell)
        # mlstm_cell_fw = tf.contrib.rnn.MultiRNNCell(cells_fw)
        # mlstm_cell_bw = tf.contrib.rnn.MultiRNNCell(cells_bw)
        # outputs_cn, _, _ = tf.contrib.rnn.static_bidirectional_rnn(mlstm_cell_fw, mlstm_cell_bw,
        #                                      x_cn, dtype=tf.float32, sequence_length=seqlen)
        cells = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cells.append(cell)
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(cells)
        outputs_cn, states = tf.contrib.rnn.static_rnn(mlstm_cell, x_cn, dtype=tf.float32,
                                                    sequence_length=seqlen)
    outputs_cn = tf.stack(outputs_cn)
    outputs_cn = tf.transpose(outputs_cn, [1, 0, 2])
    batch_size_cn = tf.shape(outputs_cn)[0]
    index_cn = tf.range(0, batch_size_cn) * timesteps + (seqlen - 1)
    outputs_cn = tf.gather(tf.reshape(outputs_cn, [-1, n_hidden]), index_cn)
    pred_cn = tf.matmul(outputs_cn, weights['cn']) + biases['cn']

    # dynamicRNN_6dof
    x_6dof = tf.unstack(x_6dof, timesteps, 1)
    with tf.variable_scope('6dof'):
        # cells_fw = []
        # cells_bw = []
        # for _ in range(num_layers):
        #     cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        #     cells_fw.append(cell)
        #     cells_bw.append(cell)
        # mlstm_cell_fw = tf.contrib.rnn.MultiRNNCell(cells_fw)
        # mlstm_cell_bw = tf.contrib.rnn.MultiRNNCell(cells_bw)
        # outputs_6dof, _, _ = tf.contrib.rnn.static_bidirectional_rnn(mlstm_cell_fw, mlstm_cell_bw,
        #                                          x_6dof, dtype=tf.float32, sequence_length=seqlen)
        cells = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cells.append(cell)
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(cells)
        outputs_6dof, _ = tf.contrib.rnn.static_rnn(mlstm_cell, x_6dof, dtype=tf.float32,
                                                       sequence_length=seqlen)
    outputs_6dof = tf.stack(outputs_6dof)
    outputs_6dof = tf.transpose(outputs_6dof, [1, 0, 2])
    batch_size_6dof = tf.shape(outputs_6dof)[0]
    index_6dof = tf.range(0, batch_size_6dof) * timesteps + (seqlen - 1)
    outputs_6dof = tf.gather(tf.reshape(outputs_6dof, [-1, n_hidden]), index_6dof)
    pred_6dof = tf.matmul(outputs_6dof, weights['6dof']) + biases['6dof']

    # staticRNN_stack
    x_stack = tf.stack([pred_6dof, pred_cn], axis=2)
    x_unstack = tf.unstack(x_stack, n_hidden, 1)
    with tf.variable_scope('stack'):
        cells_fw = []
        cells_bw = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cells_fw.append(cell)
            cells_bw.append(cell)
        mlstm_cell_fw = tf.contrib.rnn.MultiRNNCell(cells_fw)
        mlstm_cell_bw = tf.contrib.rnn.MultiRNNCell(cells_bw)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(mlstm_cell_fw, mlstm_cell_bw,
                                                                    x_unstack, dtype=tf.float32)
        # cells = []
        # for _ in range(num_layers):
        #     cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        #     cells.append(cell)
        # mlstm_cell = tf.contrib.rnn.MultiRNNCell(cells)
        # outputs, _ = tf.contrib.rnn.static_rnn(mlstm_cell, x_unstack, dtype=tf.float32)
    pred = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return pred

# prepare data
featurePath_cn_train = './features_hyperface_train.txt'
featurePath_cn_test = './features_hyperface_test.txt'
_, train_len_cn1 = loadData(featurePath_cn_train, max_seq_len_cn)
X_test_cn, test_len_cn = loadData(featurePath_cn_test, max_seq_len_cn)
featurePath_6dof_train = './features_6Dof_train.txt'
featurePath_6dof_test = './features_6Dof_test.txt'
_, train_len_6dof1 = loadData(featurePath_6dof_train, max_seq_len_6dof)
X_test_6dof, test_len_6dof = loadData(featurePath_6dof_test, max_seq_len_6dof)

y_train = np.loadtxt('./scores_train.txt')
y_test = np.loadtxt('./scores_test.txt')

batch_all = int(303*times/64)
print("times:", times)
train_len_cn = []
train_len_6dof = []
for i in range(times):
    train_len_cn = train_len_cn + train_len_cn1
    train_len_6dof = train_len_6dof + train_len_6dof1

X_train_cn = np.loadtxt("./features_hyperface_train_augmentation.txt")
X_train_6dof = np.loadtxt("./features_6Dof_train_augmentation.txt")

X_train_cn = X_train_cn[0:times*303, :]
X_train_6dof = X_train_6dof[0:times*303, :]
print(X_train_cn.shape)
print(X_train_6dof.shape)

# loss function
pred = dynamicRNN_stack(x_cn, x_6dof, seqlen, weights, biases, keep_prob, num_layers)
cost = tf.reduce_mean(tf.square(pred - y))
cost_L1 = tf.reduce_mean(tf.abs(pred - y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    test_data_cn = X_test_cn.reshape((X_test_cn.shape[0], timesteps, num_input_cn))
    test_data_6dof = X_test_6dof.reshape((X_test_6dof.shape[0], timesteps, num_input_6dof))
    test_label = y_test.reshape(-1, 1)
    test_seqlen = test_len_6dof
    for i in range(len(test_seqlen)):
        test_seqlen[i] /= num_input_6dof

    for ep in range(0, epoch):
        totalNum = X_train_cn.shape[0]
        indices = [x for x in range(totalNum)]
        random.shuffle(indices)
        train_loss = 0.0

        for step in range(0, batch_all+1):
            batch_x_cn, batch_y, batch_seqlen, btO = gen_next_batch_cn(X_train_cn, y_train, train_len_cn, batch_size, indices, step)
            batch_x_6dof, _, _ = gen_next_batch_6dof(X_train_6dof, y_train, train_len_6dof, batch_size, indices, step)
            batch_x_cn = batch_x_cn.reshape((btO, timesteps, num_input_cn))
            batch_x_6dof = batch_x_6dof.reshape((btO, timesteps, num_input_6dof))
            batch_y = batch_y.reshape(-1, 1)
            for i in range(batch_seqlen.shape[0]):
                batch_seqlen[i] /= num_input_cn

            sess.run(optimizer, feed_dict={x_cn: batch_x_cn, x_6dof: batch_x_6dof, y: batch_y,
                                           seqlen: batch_seqlen, drop_keep_rate: keep_prob})
            train_loss_tmp = sess.run(cost, feed_dict={x_cn: batch_x_cn, x_6dof: batch_x_6dof, y: batch_y,
                                             seqlen: batch_seqlen, drop_keep_rate: 1.0})
            train_loss = train_loss + train_loss_tmp
