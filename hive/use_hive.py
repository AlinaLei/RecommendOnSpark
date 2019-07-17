from pyspark.sql import HiveContext
from pyspark.sql import SparkSession,Row
import os

from config import config

class HiveOperate():
    def __init__(self,sc):
        """
        创建Hive session
        :return:
        """
        #self.sc = config.CreateSparkContext()
        self.hive_context = HiveContext(sc)
        return self.hive_context

    def df_insert_to_hive(self, df, table_name='channel_result', database='sparktest'):
        """
        将数据插入到hive中
        :param df:
        :param table_name:
        :param database:
        :return:
        """
        df.registerTempTable("result_tmp")
        result1 = self.hive_context.sql("select * from result_tmp limit 10")
        result1.show()
        self.hive_context.sql("use {}".format(database))
        self.hive_context.sql("drop table if EXISTS  {} ".format(table_name))
        self.hive_context.sql("create table {} as select * from result_tmp where 1 = 2 ".format(table_name))
        self.hive_context.sql(" insert overwrite table {} select * from result_tmp".format(table_name))