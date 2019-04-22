import numpy as np
import tensorflow as tf
import csv
import re
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt


HIDDEN_SIZE=30
NUM_LAYERS=2

TIMESTEPS=6
TRAINING_STEPS=40000
BATCH_SIZE=20

TRAINING_EXAMPLES=10000
TESTING_EXAMPLES=500#2500
SAMPLE_GAP=0.01
MODEL_SAVE_PATH="./"
MODE_NAME="model.ckpt"
def generate_data(seq):
    x=[]
    y=[]
    for i in range(len(seq)-TIMESTEPS):
        x.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(X,y,is_traing):
    cell=tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
                                      for _ in range(NUM_LAYERS)])
    output,_=tf.nn.dynamic_rnn(cell,X,dtype=np.float32)
    output=output[:,-1,:]

    predictions=tf.contrib.layers.fully_connected(output,1,activation_fn=None)
    if not is_traing:
        return predictions,None,None

    loss=tf.losses.mean_squared_error(labels=y,predictions=predictions)

    train_op=tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),optimizer="Adagrad",learning_rate=0.001)
    return predictions,loss,train_op
def train(sess,train_x,train_y):
    #ds=tf.data.Dataset.from_tensor_slices((train_x,train_y))
    #ds=ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    #X,y=ds.make_one_shot_iterator().get_next()
    #X=train_x
    #y=train_y
    X = tf.placeholder(dtype=tf.float32,shape=(BATCH_SIZE,TIMESTEPS,6),name="input_placeholder")
    y = tf.placeholder(dtype=tf.float32,shape=(BATCH_SIZE,1),name="pred_placeholder")
    #X=train_x[0]
    #y=train_y[0]
    with tf.variable_scope("model"):
        predication,loss,train_op=lstm_model(X,y,True)
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

def run_eval1(sess,test_X,test_y):
    X = test_X
    y = tf.constant(test_y, tf.float32)
    with tf.variable_scope("model", reuse=True):
        prediction, _, _, = lstm_model(X, [0.0], False)
    predictions = []
    labels = []
    #for i in range(TESTING_EXAMPLES):
    p, l = sess.run([prediction, y])
        #predictions.append(p)
        #labels.append(l)
    #predictions = np.array(predictions[0]).squeeze()
    #labels = np.array(labels[0]).squeeze()
    #rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print(p)
    print(l)
    #print("Mean Square Error is :%f" % rmse)
def run_eval(sess,test_X,test_y):
    
    #ds=tf.data.Dataset.from_tensor_slices((test_X,test_y))
    #ds=ds.batch(1)
    #X,y=ds.make_one_shot_iterator().get_next()
    X=test_X
    y = tf.constant(test_y, tf.float32)
    #y=test_y
    with tf.variable_scope("model",reuse=True):
        prediction,_,_,=lstm_model(X,[0.0],False)
    predictions=[]
    labels=[]
    for i in range(TESTING_EXAMPLES):
        p,l=sess.run([prediction,y])
        predictions.append(p)
        labels.append(l)
    predictions=np.array(predictions[0]).squeeze()
    labels=np.array(labels[0]).squeeze()
    rmse=np.sqrt(((predictions-labels)**2).mean(axis=0))
    print(predictions)
    print(labels)
    print("Mean Square Error is :%f" % rmse)

    plt.figure()
    plt.plot(predictions,label='predictions')
    plt.plot(labels,label='real_sin')
    plt.legend()
    plt.show()

test_start=(TRAINING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
test_end=test_start+(TESTING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
train_X,train_y=generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES+TIMESTEPS,dtype=np.float32)))
test_X,test_Y=generate_data(np.sin(np.linspace(test_start,test_end,TESTING_EXAMPLES+TIMESTEPS,dtype=np.float32)))
def dataSet():
    csv_file = csv.reader(open('monthlstmDataSet.csv', 'r'))
    Train_X = []
    Train_y = []
    Test_X = []
    Test_y = []
    n = 0
    for i in csv_file:
        if n < 1400:
            count_t = 0
            tmp = []
            tmpAll = []
            if( re.match(r'(.*?)06(.*?)',i[1]) and float(i[-3])<5 and float(i[-3])>-1):
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
                n+=1
        elif n < 1600:
            count_t = 0
            tmp = []
            tmpAll = []
            if (re.match(r'(.*?)06(.*?)',i[1]) and float(i[-3]) < 5 and float(i[-3]) > -1):
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
    count_n=0
    tmpXBatchsize=[]
    tmpyBatchsize=[]
    Train_Xafter=[]
    Train_yafter=[]
    Test_Xafter=[]
    Test_yafter=[]
    for i in range(1200):
        if(count_n<BATCH_SIZE):
            tmpXBatchsize.append(Train_X[i])
            tmpyBatchsize.append(Train_y[i])
        elif count_n==BATCH_SIZE:
            Train_Xafter.append(tmpXBatchsize)
            Train_yafter.append(tmpyBatchsize)
            tmpXBatchsize=[]
            tmpyBatchsize=[]
            tmpXBatchsize.append(Train_X[i])
            tmpyBatchsize.append(Train_y[i])
            count_n=0
        count_n+=1
    #Train_Xafter.append(tmpXBatchsize)
    #Train_yafter.append(tmpyBatchsize)
    tmpXBatchsize = []
    tmpyBatchsize = []
    print(Train_X)
    print(Train_y)
    """
    count_n=0
    for i in range(500):
        if(count_n<BATCH_SIZE):
            tmpXBatchsize.append(Test_X[i])
            tmpyBatchsize.append(Test_y[i])
        elif count_n==BATCH_SIZE:
            Test_Xafter.append(tmpXBatchsize)
            Test_yafter.append(tmpyBatchsize)
            tmpXBatchsize=[]
            tmpyBatchsize=[]
            tmpXBatchsize.append(Test_X[i])
            tmpyBatchsize.append(Test_y[i])
            count_n=0
        count_n+=1
    Test_Xafter.append(tmpXBatchsize)
    Test_yafter.append(tmpyBatchsize)
    """
    return Train_Xafter,Train_yafter,Test_X,Test_y#Test_Xafter,Test_yafter
    """
    return Train_X,Train_y,Test_X,Test_y
    """
with tf.Session() as sess:
    Train_X, Train_y, Test_X, Test_y=dataSet()
    Train_X=np.array(Train_X,dtype=np.float32)
    Train_y=np.array(Train_y,dtype=np.float32)
    print(len(Train_X))
    print(len(Test_X))
    Test_X=np.array(Test_X,dtype=np.float32)
    Test_y=np.array(Test_y,dtype=np.float32)
    print(Train_X.shape)
    print(Train_y.shape)
    #seq_test = np.sin(np.linspace(start=100, stop=110, num=11000, dtype=np.float32))
    #test1_x,test1_y=generate_data(seq_test)
    #test1_x= np.reshape(test1_x, newshape=(-1, 10, 1))
    train(sess,Train_X,Train_y)
    run_eval(sess,Test_X,Test_y)
    #run_eval1(sess,Test_X[0],Test_y[0])
    #run_eval(sess,Train_X,Train_y)


