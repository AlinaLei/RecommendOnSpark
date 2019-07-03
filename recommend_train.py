#/usr/bin/env python3
# -*-coding:utf:8 -*-

from pyspark.sql import SparkSession,Row
from pyspark.sql import HiveContext
from pyspark.mllib.recommendation import ALS, Rating, MatrixFactorizationModel
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import DenseVector
import os
#os.environ["PYSPARK_PYTHON"]="/usr/local/python3"  # set python version
def CreateSparkContext():
    # 构建SparkSession实例对象
    spark = SparkSession.builder \
        .appName("TestSparkSession") \
        .master("spark://hadoop2:7077") \
        .config("hive.metastore.uris", "thrift://hadoop1:9083") \
        .config('spark.executor.num','4')\
        .config('spark.executor.memory','64g')\
        .config("spark.executor.cores",'4')\
        .config('spark.cores.max','16')\
        .config('spark.driver.memory','32g') \
        .config("spark.sql.catalogImplementation", "hive") \
        .getOrCreate()

    # 获取SparkContext实例对象
    sc = spark.sparkContext
    return sc

def CreateSparkContext_tmp():
    spark = SparkSession.builder \
        .appName("SparkSessionExample") \
        .master("local") \
        .getOrCreate()

    # 获取SparkContext实例对象
    sc = spark.sparkContext
    return sc

def sc_path(pathtype,path):
    global Path
    if pathtype == "local":
        Path = "file://"+path

    else:
        Path = "hdfs://hadoop2:9000/root/hadoop/input/data/"
    print("the path is :{}".format(Path))
    return Path

def read_file_to_RDD(sc, path,pathtype="local"):
    """
    读取文件到RDD
    :param sc:
    :param path:
    :param pathtype:
    :return:
    """
    return sc.textFile(sc_path(pathtype,path))

def transform_rdd_to_DF(rdd, columns_list):
    """
    将RDD类型转换为DataFrame类型
    :param rdd:
    :param columns_list:
    :return:
    """
    df = rdd.toDF(columns_list)
    return df

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

def hadle_result(line):
    """
    :param RDD: 这里的RDD是一条数据，长这样 (451, (Rating(user=451, product=1426, rating8=.297368554401814),))
    :return:
    """
    """#如果用户与产品之间的关系是一对多的话就需要涉及到
    product = []
    rating = []
    for item in line[1]:
        product.append(item[1])
        rating.append(item[2]) """
    for item in line[1]:
        product= item[1]
        rating = item[2]
    return Row (user=line[0],products =product ,rating =rating)

def handle_DataFrame(df1,df2,col_name):
    """
    拼接两个DataFrame
    :param df1:
    :param df2:
    :param col_name: 拼接的Key
    :return:
    """
    result_df = df1.join(df2,[col_name],"left")
    return result_df

def save_DF(df, path, sep="|",pathtype="local"):
    """
    保存DataFrame类型的数据到文件
    :param df:
    :param path:
    :param sep:
    :return:
    """
    # 将df保存输出的时候coalesce(1)的意思就是将输出到文件都放在一起而不进行拆分，如果不指定在大数据量的情况下文件输出会自动拆分
    df.coalesce(1).write.csv(path=sc_path(pathtype,path), header=False, sep=sep, mode='overwrite')

class HiveOperator():
    def __init__(self, sc):
        self.sc = sc
        return HiveContext(self.sc)

    def result_to_hive(self,sql_list):
        hive_context = HiveContext(self.sc)
        hive_context = hive_context(sc)
        return hive_context.spl(sql_list)

if __name__ == "__main__":

    #训练模型
    sc = CreateSparkContext()
    raw_ratings_rdd =read_file_to_RDD(sc,"/data/lin/train_data/user_data/part-00000-fa8d558c-15be-4399-a575-f0a5391c46f9-c000.csv")
    ratings_rdd = handle_data(raw_ratings_rdd,3)
    try:
        ratings_datas = create_als_data(ratings_rdd)
        training_ratings, testing_ratings = split_train_test_data(ratings_datas)
        model, rmse_value, rank, iterations, lambda_=train_model_evaluate(training_ratings, testing_ratings, 10, 4, 0.0001)
        try:
            #save_model(sc, model, "file:///data/lin/savemodel/als_model_test")

            print("start recommned result")
            recommend = Recommend(ALS_model=model)
            recommendation_all = recommend.map(hadle_result).toDF()
            category = read_file_to_RDD(sc, "/data/lin/train_data/user_data/category.txt")
            catrgory_rdd = handle_data(category, 3,sep =',')
            category_df = transform_rdd_to_DF(catrgory_rdd, ['products','category','channel'])
            result = handle_DataFrame(recommendation_all, category_df,'products')
            print("the result head is :{}".format(result.head(4)))
            save_DF(result.rdd.map(lambda l:Row(str(l.user)+"|"+str(l.products)+"|"+str(l.rating)+"|"+str(l.category)+"|"+str(l.channel))).toDF(), "/data/lin/predict_data/recommend_movie_result/test")
            #result.toDF(['user','products','rating','category','channel']).registerTempTable("result_tmp")

            try:
                #Hive相关操作
                result_tmp=sc.textFile("/data/lin/predict_data/recommend_movie_result/test").map(lambda line: line.split("|")[0:5]).toDF(['user','products','rating','category','channel']).registerTempTable("result_tmp")
                hive_context = HiveContext(sc)
                hive_result = hive_context.sql( """select * from result_tmp limit 10""")
                hive_result.show()
                hive_context.sql("""use sparktest""")
                hive_context.sql("""drop table if EXISTS  recommend_result """)
                hive_context.sql("""create table recommend_result as select * from result_tmp where 1=2 """)
                hive_context.sql("""insert overwrite  table recommend_result  select * from result_tmp""")

            except Exception as e:
                print(str(e))
                print("insert into hive failed")
        except Exception as e:
            print(str(e))
            print("save model failed")
    except Exception as e:
        print(str(e))
    """
    #加载模型
    sc = CreateSparkContext()
    try:
        recommend =LoadModel(sc, "file:///data/lin/savemodel/als_model_test")
        recommendation_all = recommend.map(hadle_result).toDF()
        print("the recommendation is :{}".format(recommendation_all.head(3)))
        category = read_file_to_RDD(sc, "/data/lin/train_data/user_data/category.txt")
        catrgory_rdd = handle_data(category, 3, sep=',')
        category_df = transform_rdd_to_DF(catrgory_rdd, ['products', 'category', 'channel'])
        result = handle_DataFrame(recommendation_all, category_df, 'products')
        save_DF(result, "file:///data/lin/predict_data/recommend_movie_result/test")

    except Exception as e:
        print(str(e))
        print("recommend failed")
    """



