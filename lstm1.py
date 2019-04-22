import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import csv
import re
import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt

HIDDEN_SIZE = 20
NUM_LAYERS = 2

TIMESTEPS = 120
TRAINING_STEPS = 5000
BATCH_SIZE = 20

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 500  # 2500
SAMPLE_GAP = 0.01
MODEL_SAVE_PATH = "./"
MODE_NAME = "model.ckpt"


def generate_data(seq):
    x = []
    y = []
    for i in range(len(seq) - TIMESTEPS):
        x.append([seq[i:i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y, is_traing):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
                                        for _ in range(NUM_LAYERS)])
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=np.float32)
    output = output[:, -1, :]

    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    if not is_traing:
        return predictions, None, None

    #loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    loss=tf.reduce_mean(tf.abs(y-predictions))

    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer="Adagrad", learning_rate=0.001)
    return predictions, loss, train_op


def train(sess, train_x, train_y):
    # ds=tf.data.Dataset.from_tensor_slices((train_x,train_y))
    # ds=ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    # X,y=ds.make_one_shot_iterator().get_next()
    # X=train_x
    # y=train_y
    # X = tf.placeholder(dtype=tf.float32,shape=(None,TIMESTEPS,1),name="input_placeholder")
    # y = tf.placeholder(dtype=tf.float32,shape=(None,1),name="pred_placeholder")
    X = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, TIMESTEPS, 6), name="input_placeholder")
    y = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 1), name="pred_placeholder")
    with tf.variable_scope("model"):
        predication, loss, train_op = lstm_model(X, y, True)
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        if i<len(train_x):
            iR=i
        else:
            iR=i%len(train_x)
        xs=train_x[iR]
        ys=train_y[iR]
        _,l=sess.run([train_op,loss],feed_dict={X:xs,y:ys})
        if i%100==0:
            print("train step:"+str(i)+" ,loss:"+str(l))


def run_eval(sess, test_X, test_y):
    # ds=tf.data.Dataset.from_tensor_slices((test_X,test_y))
    # ds=ds.batch(1)
    # X,y=ds.make_one_shot_iterator().get_next()
    X = test_X
    y = tf.constant(test_y, tf.float32)
    # y=test_y
    with tf.variable_scope("model", reuse=True):
        prediction, _, _, = lstm_model(X, [0.0], False)
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)
    predictions = np.array(predictions[0]).squeeze()
    labels = np.array(labels[0]).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print(predictions)
    print(labels)
    print("Mean Square Error is :%f" % rmse)

    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real')
    plt.legend()
    plt.show()


test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_X, test_Y = generate_data(
    np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

def lstmDataSet():
    csv_file = csv.reader(open('dayDataAfter.csv', 'r'))
    y=[]
    X=[]
    for i in csv_file:
        tmpData1 = re.findall(r"[\'](.*?)[\']", i[0])
        y.append([float(tmpData1[2])])
        tmp=[]
        for j in range(1,len(i)):
            tmpSecond=[]
            tmpData = re.findall(r"[\'](.*?)[\']", i[j])
            for z in range(1,len(tmpData)-2):
                tmpSecond.append(float(tmpData[z]))
            #tmpSecond.append(float(tmpData[-2])/1000.0)
            #tmpSecond.append(float(tmpData[-1]) / 100000.0)
            tmp.append(tmpSecond)
        X.append(tmp)
    xAfter=[]
    yAfter=[]
    for i in range(0,len(X)):
        flag=0
        if y[i]==float('inf') or y[i]==float('nan') or y[i][0]>2 or y[i][0]<-1:
            continue
        for j in X[i]:
            for z in j:
                if z == float('inf') or z == float('nan'):
                    flag=1
        if flag==0:
            xAfter.append(X[i])
            yAfter.append(y[i])
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    for i in range(0, 2300):
        train_X.append(xAfter[i])
        train_y.append(yAfter[i])
    for i in range(2300, 2700):
        test_X.append(xAfter[i])
        test_y.append(yAfter[i])
        count_n = 0
        tmpXBatchsize = []
        tmpyBatchsize = []
        Train_Xafter = []
        Train_yafter = []
    for i in range(2300):
        if(count_n<BATCH_SIZE):
            tmpXBatchsize.append(train_X[i])
            tmpyBatchsize.append(train_y[i])
        elif count_n==BATCH_SIZE:
            Train_Xafter.append(tmpXBatchsize)
            Train_yafter.append(tmpyBatchsize)
            tmpXBatchsize=[]
            tmpyBatchsize=[]
            tmpXBatchsize.append(train_X[i])
            tmpyBatchsize.append(train_y[i])
            count_n=0
        count_n+=1
    return Train_Xafter,Train_yafter,test_X,test_y

def dataSet():
    csv_file = csv.reader(open('monthlstmDataSet.csv', 'r'))
    Train_X = []
    Train_y = []
    Test_X = []
    Test_y = []
    n = 0
    for i in csv_file:
        if n < 2900:
            count_t = 0
            tmp = []
            tmpAll = []
            if (float(i[-3]) < 10):
                for j in range(2, 38):
                    if count_t < 6:
                        tmp.append(float(i[j]))
                    else:
                        tmpAll.append(tmp)
                        tmp = []
                        tmp.append(float(i[j]))
                        count_t = 0
                    count_t += 1
                tmpAll.append(tmp)
                Train_X.append(tmpAll)
                Train_y.append([float(i[-3])])
                n += 1
        elif n < 3400:
            count_t = 0
            tmp = []
            tmpAll = []
            if (float(i[-3]) < 10):
                for j in range(2, 38):
                    if count_t < 6:
                        tmp.append(float(i[j]))
                    else:
                        tmpAll.append(tmp)
                        tmp = []
                        tmp.append(float(i[j]))
                        count_t = 0
                    count_t += 1
                Test_X.append(tmpAll)
                Test_y.append([float(i[-3])])
                n += 1
    return Train_X, Train_y, Test_X, Test_y


with tf.Session() as sess:
    Train_X, Train_y, Test_X, Test_y = lstmDataSet()
    Train_X = np.array(Train_X, dtype=np.float32)
    Train_y = np.array(Train_y, dtype=np.float32)
    print(len(Train_X))
    print(len(Test_X))
    Test_X = np.array(Test_X, dtype=np.float32)
    Test_y = np.array(Test_y, dtype=np.float32)
    train(sess, Train_X, Train_y)
    run_eval(sess, Test_X, Test_y)
    # run_eval(sess,Train_X,Train_y)


