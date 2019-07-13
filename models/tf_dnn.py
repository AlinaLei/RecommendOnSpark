import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense ,Activation ,BatchNormalization
from keras.utils import np_utils
from config import config
from data_feature import data_handle
from models import model_feature

batch_size = 500
nb_classes = 20
nb_epoch = 10
ratio = 0.9

def train_dnn():
    model = Sequential()
    model.add(Dense(64, input_dim=2))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(20, activation='sigmoid'))

def handle_data(train_data_path):
    sc = config.CreateSparkContext()
    raw_ratings_rdd = data_handle.read_file_to_RDD(sc, train_data_path, pathtype='local')
    print("start read data test 2")
    ratings_rdd = model_feature.handle_read_data(raw_ratings_rdd, 3)
    ratings_df = data_handle.transform_rdd_to_DF(ratings_rdd,['user_id','products_id','rating'])
    len = ratings_df.count()
    end_size = int(len*ratio)
    X = ratings_df.select(['user_id','rating']).collect()
    Y = ratings_df.select("products_id").collect()
    x_train = X[:end_size]
    x_test = X[end_size:]
    y_train = Y[:end_size]
    y_test = Y[end_size:]
    return x_train,x_test,y_train,y_test

if __name__=='__main__':
    x_train, x_test, y_train, y_test =handle_data("/data/lin/train_data/user_data/part-00000-fa8d558c-15be-4399-a575-f0a5391c46f9-c000.csv")
    model = train_dnn()
    model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.fit(x_train,y_train,batch_size = batch_size ,epochs=nb_epoch )
    loss_and_metrics = model.evaluate(x_test,y_test,batch_size=batch_size)
    print("the loss is :{}".format(loss_and_metrics))



