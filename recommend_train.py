#/usr/bin/env python3
# -*-coding:utf:8 -*-

from models import model_feature
from config.config import *
from data import data_handle



##TODO 拆分结果集
##TODO 将拆分结果保存到文件中
##TODO 重命名文件名
"""
            try:
                #Hive相关操作
                result_tmp=sc.textFile("/data/lin/predict_data/recommend_movie_result/test/*").map(lambda line: line.split("|")[0:5]).toDF(['user','products','rating','category','channel']).registerTempTable("result_tmp")
                hive_context = HiveContext(sc)
                hive_result = hive_context.sql( ""select * from result_tmp limit 10"")
                hive_result.show()
                hive_context.sql(""use sparktest"")
                hive_context.sql(""drop table if EXISTS  recommend_result "")
                hive_context.sql(""create table recommend_result as select * from result_tmp where 1=2 "")
                hive_context.sql(""insert overwrite  table recommend_result  select * from result_tmp"")

            except Exception as e:
                print(str(e))
                print("insert into hive failed") """

if __name__ == "__main__":
    #训练模型
    sc = CreateSparkContext()
    print("start test1")
    raw_ratings_rdd = data_handle.read_file_to_RDD(sc, "/data/lin/train_data/user_data/part-00000-fa8d558c-15be-4399-a575-f0a5391c46f9-c000.csv")
    print("start read data test 2")
    ratings_rdd = model_feature.handle_read_data(raw_ratings_rdd,3)
    try:
        print("start handle data test 3")
        ratings_datas = model_feature.create_als_data(ratings_rdd)
        print("start split data test 4")
        training_ratings, testing_ratings = model_feature.split_train_test_data(ratings_datas)
        print("start train model test 5")
        model, rmse_value, rank, iterations, lambda_=model_feature.train_model_evaluate(training_ratings, testing_ratings, 10, 4, 0.0001)
        try:
            #save_model(sc, model, "file:///data/lin/savemodel/als_model_test")

            print("start recommned result")
            recommend = model_feature.Recommend(ALS_model=model)

            recommendation_all = recommend.map(data_handle.hadle_result).toDF()
            category = data_handle.read_file_to_RDD(sc, "/data/lin/train_data/user_data/category.txt")
            catrgory_rdd = model_feature.handle_read_data(category, 3,sep =',')
            category_df = data_handle.transform_rdd_to_DF(catrgory_rdd, ['products','category','channel'])
            result = data_handle.handle_DataFrame(recommendation_all, category_df,'products')
            data_handle.split_data_by_category(result, 'category', "/data/lin/predict_data/recommend_movie_result/test",mode='overwrite')

            print("the result head is :{}".format(result.show(4)))
            data_handle.save_DF(result, "/data/lin/predict_data/recommend_movie_result/test/category_result")
            #save_DF(result.rdd.map(lambda l:Row(str(l.user)+"|"+str(l.products)+"|"+str(l.rating)+"|"+str(l.category)+"|"+str(l.channel))).toDF(), "/data/lin/predict_data/recommend_movie_result/test")
            #result.toDF(['user','products','rating','category','channel']).registerTempTable("result_tmp")
            
            
    
            
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
        catrgory_rdd = data(category, 3, sep=',')
        category_df = transform_rdd_to_DF(catrgory_rdd, ['products', 'category', 'channel'])
        result = handle_DataFrame(recommendation_all, category_df, 'products')
        save_DF(result, "file:///data/lin/predict_data/recommend_movie_result/test")

    except Exception as e:
        print(str(e))
        print("recommend failed")
    """



