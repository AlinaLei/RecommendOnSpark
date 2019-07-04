#/usr/bin/env python3
# -*-coding:utf:8 -*-

from pyspark.sql import SparkSession,Row
from pyspark.sql import HiveContext
from pyspark.mllib.recommendation import ALS, Rating, MatrixFactorizationModel
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import DenseVector
import os

def handle_data(raw_RDD,num_list,sep="|"):
    """

    :param raw_RDD: RDD类型
    :return:  只获取前三行数据，也就是用户，产品，评分
    """
    return raw_RDD.map(lambda line: line.split(sep)[0:num_list])

def split_train_test_data(raw_RDD,rates=[0.8,0.2]):
    """
    拆分测试集以及训练集
    :param raw_RDD:
    :param rates: 拆分比列
    :return: 返回随机拆分后的训练集以及测试集
    """
    training_ratings, testing_ratings = raw_RDD.randomSplit(rates)
    return training_ratings,testing_ratings

def create_als_data(raw_RDD):
    """
    处理数据为ALS模型接受的数据类型Rating
    :param raw_RDD:
    :return:
    """
    return raw_RDD.map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2])))

def alsModelEvaluate(model, testing_rdd):
    """
    评估als模型效果
    :param model:
    :param testing_rdd:
    :return:
    """
    # 对测试数据集预测评分，针对测试数据集进行预测
    predict_rdd = model.predictAll(testing_rdd.map(lambda r: (r[0], r[1])))
    print(predict_rdd.take(5))
    predict_actual_rdd = predict_rdd.map(lambda r: ((r[0], r[1]), r[2])) \
        .join(testing_rdd.map(lambda r: ((r[0], r[1]), r[2])))

    print(predict_actual_rdd.take(5))
    # 创建评估指标实例对象
    metrics = RegressionMetrics(predict_actual_rdd.map(lambda pr: pr[1]))

    print("MSE = %s" % metrics.meanSquaredError)
    print("RMSE = %s" % metrics.rootMeanSquaredError)
    # 返回均方根误差
    return metrics.rootMeanSquaredError


def train_model_evaluate(training_rdd, testing_rdd, rank, iterations, lambda_):
    """
    训练ALS模型并评估模型
    :param training_rdd: 训练集
    :param testing_rdd: 测试集
    :param rank: 隐藏因子的个数
    :param iterations: 迭代次数
    :param lambda_: 正则项的惩罚系数
    :return:
    """
    # 定义函数，训练模型与模型评估
    # 使用超参数的值，训练数据和ALS算法训练模型
    model = ALS.train(training_rdd, rank, iterations, lambda_)

    # 模型的评估
    rmse_value = alsModelEvaluate(model, testing_rdd)

    # 返回多元组
    return (model, rmse_value, rank, iterations, lambda_)

def find_best_model(training_ratings, testing_ratings,rank_list=[10, 20],iterations_list=[10, 20],lambda_list=[0.001, 0.01]):
    """
    通过迭代的方式来找到较好的模型参数
    :param rank_list: 隐藏因子的个数的列表
    :param iterations_list: 迭代次数的列表
    :param lambda_list: 正则项的惩罚系数的列表
    :return: 在给定的列表中最优的组合
    """
    metrix_list = [
        train_model_evaluate(training_ratings, testing_ratings, param_rank, param_iterations, param_lambda)
        for param_rank in rank_list
        for param_iterations in iterations_list
        for param_lambda in lambda_list
        ]
    sorted(metrix_list, key=lambda k: k[1], reverse=False)
    model, rmse_value, rank, iterations, lambda_ = metrix_list[0]
    print("The best parameters is  rank={}, iterations={}, lambda_={}".format(rank, iterations, lambda_))
    return rank, iterations, lambda_

def save_model(sc,model,path):
    """
    保存模型
    :param sc:
    :param model:
    :param path: 保存模型路径
    :return:
    """
    model.save(sc, path)
    print("model has been saved")

def LoadModel(sc, path):
    try:
        ALS_model = MatrixFactorizationModel.load(sc, path)
        print("载入模型成功")
        return ALS_model
    except Exception:
        print("模型不存在，请先训练模型")
        return None

def Recommend(ALS_model, type_for='U', k=1):
    """
    生成推荐结果
    :param ALS_model:
    :param type_for: 基于用户还是基于产品类型
    :param k: 推荐个数，这里默认是一对一
    :return:
    """
    if type_for == "U":
        result = ALS_model.recommendProductsForUsers(k)
        return result
    if type_for == "P":
        result = ALS_model.recommendUsersForProducts(k)
        return result