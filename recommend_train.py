#/usr/bin/env python3
# -*-coding:utf:8 -*-
import os
from models import train_model , tf_dnn

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
    os.system("rm -rf data_feature.zip")
    os.system("rm -rf models.zip")
    os.system("rm -rf config.zip")
    os.system("rm -rf hive.zip")
    os.system("zip -r -0 models.zip models/")
    os.system("zip -r -0 config.zip config/")
    os.system("zip -r -0 data_feature.zip data_feature/")
    os.system("zip -r -0 hive.zip hive/")
    train_model.train_model_feature("/data/lin/train_data/user_data/part-00000-fa8d558c-15be-4399-a575-f0a5391c46f9-c000.csv","/data/lin/train_data/user_data/category.txt")
    #tf_dnn.tf_workflow("/data/lin/train_data/user_data/part-00000-fa8d558c-15be-4399-a575-f0a5391c46f9-c000.csv")


