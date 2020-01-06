import pickle as pkl
import tensorflow as tf
import numpy as np


def shuffle(arr):
    return np.random.permutation(arr)


def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


dataset_path = 'train.pkl'


train_pkl = pkl.load(open(dataset_path, 'rb')) 

X_data = train_pkl[0]
Y_data = train_pkl[1]

Y_data = np.reshape(Y_data, Y_data.shape[0])

X_data, Y_data = next_batch(X_data.shape[0], X_data, Y_data)

idx_divide = int(X_data.shape[0]*0.8)

X_train = X_data[:idx_divide,:]
y_train = Y_data[:idx_divide]

X_test = X_data[idx_divide:,:]
y_test = Y_data[idx_divide:]


input_l_size = 3136
hidden_l_size = 220
output_l_size = 36
batch_size = 100
num_epochs = 1000
learning_rate = 0.000095
regularization_coeff = 0.00165

def params():
    hidden_size = int(random.uniform(180, 240))
    batch_size = int(random.uniform(64, 128))
    learning_rate = 10**random.uniform(-4,-2)
    reg_lambda = 10**random.uniform(-4,-3)
    return (hidden_size, batch_size, learning_rate, reg_lambda)

percentage_correct = lambda y, y_hat: 100. * np.mean(y == y_hat)


def predict(X, y_pred, session):
    prediction = tf.argmax(y_pred, 1)
    return (prediction.eval(feed_dict={x: X}, session=session)).reshape(X.shape[0])

def train(iter_num, X_train, y_train):
    with tf.Session() as session:
        session.run(init_op)
        try:
            best_acc = 0
            acc = 0
            num_nonimprovments = 0
            for i in range(iter_num):
                avg_cost = 0
                idx = np.random.permutation(X_train.shape[0])
                X_train = X_train[idx, :]
                y_train = y_train[idx]
                for j in range(0, X_train.shape[0], batch_size):
                    bs = min(X_train.shape[0] - j, batch_size)
                    batch_xs, batch_true_ys = X_train[j:j + bs, :], y_train[j:j + bs]
                    feed_dict_train = {x: batch_xs, y_true: batch_true_ys}
                    session.run(train_step, feed_dict=feed_dict_train)
                    curr_cost = session.run(cost, feed_dict=feed_dict_train)
                    avg_cost = avg_cost + curr_cost

                print("epoch: {}, avg. cost: {}".format(i, avg_cost / (X_train.shape[0] / batch_size)))
                if i % 5 == 0:
                    num_nonimprovments += 1
                    y_hat = predict(X_test, y_pred, session)
                    y_hat1 = predict(X_train, y_pred, session)
                    acc = percentage_correct(y_test, y_hat)
                    acc1 = percentage_correct(y_train, y_hat1)
                    print("test acc: {}, train acc: {}".format(acc, acc1))
                    if acc > best_acc:
                        num_nonimprovments = 0
                        print("Save!")
                        best_acc = acc
                        bestb = biases.eval()
                        bestw = weights.eval()
                        bestb1 = biases1.eval()
                        bestw1 = weights1.eval()

                    if num_nonimprovments > 6:
                        break

        except KeyboardInterrupt:
            return (bestw, bestb, bestw1, bestb1, best_acc)

        if acc > best_acc:
            bestb = biases.eval()
            bestw = weights.eval()
            bestb1 = biases1.eval()
            bestw1 = weights1.eval()

        return (bestw, bestb, bestw1, bestb1, best_acc)



for i in range (100):

    hidden_l_size, batch_size,learning_rate,regularization_coeff = params()

    print("neurons: {}; batch: {}; learning: {}; reg_coeff: {}".format(hidden_l_size, batch_size,learning_rate,regularization_coeff))

    x = tf.placeholder(tf.float32,[None,input_l_size])

    weights = tf.Variable(tf.random_uniform([input_l_size, hidden_l_size]) * tf.sqrt(2 / input_l_size))
    biases = tf.Variable(tf.zeros([hidden_l_size]))

    weights1 = tf.Variable(tf.random_uniform([hidden_l_size,output_l_size]) * tf.sqrt(2 / hidden_l_size))
    biases1 = tf.Variable(tf.zeros([output_l_size]))

    hiddenlayer = tf.add(tf.matmul(x,weights), biases)
    hiddenlayer = tf.nn.relu(hiddenlayer)


    logits = tf.matmul(hiddenlayer,weights1) + biases1


    y_pred = tf.nn.softmax(logits)
    y_true = tf.placeholder(tf.int32,[None,])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    regularizers = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights1)
    cost = tf.reduce_mean(cost + regularization_coeff * regularizers)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init_op = tf.global_variables_initializer()

    best_weights,best_biases,best_weights1,best_biases1, best_acc = train(num_epochs, X_train, y_train)
    pkl.dump((best_weights, best_biases, best_weights1, best_biases1), open("model-%.2f%%.pkl" % best_acc, "wb"))
    pkl.dump((hidden_l_size, batch_size,learning_rate,regularization_coeff), open("model-%.2f%%-hyperparams.pkl" % best_acc, "wb"))




