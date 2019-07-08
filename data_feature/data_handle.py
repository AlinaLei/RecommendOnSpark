from config import config
from pyspark.sql import Row
import pyspark.sql.types as typ
def read_file_to_RDD(sc, path,pathtype="local"):
    """
    读取文件到RDD
    :param sc:
    :param path:
    :param pathtype:
    :return:
    """
    return sc.textFile(config.sc_path(pathtype,path))

def transform_rdd_to_DF(rdd, columns_list):
    """
    将RDD类型转换为DataFrame类型
    :param rdd:
    :param columns_list:
    :return:
    """
    df = rdd.toDF(columns_list)
    return df

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

def read_csv_to_df(sc,path,header=False):
    """
    直接读取csv文件到DataFrame
    :param sc:
    :param path:
    :param header:
    :return:
    """
    labels = [('user_id',typ.IntegerType()),
              ('product_id',typ.IntegerType()),
              ('rating',typ.FloatType())
              ]
    schema = typ.StructType([
        typ.StructField(e[0],e[1],False) for e in labels
    ])
    user_data =sc.read.csv(path,header=header,schema=schema)
    return user_data

def save_DF(df, path,pathtype="local"):
    """
    保存DataFrame类型的数据到文件
    :param df:
    :param path:
    :param sep:
    :return:
    """
    if pathtype == "local":
        Path = "file://"+path

    else:
        Path = "hdfs://hadoop2:9000/root/hadoop/input/data/"
    # 将df保存输出的时候coalesce(1)的意思就是将输出到文件都放在一起而不进行拆分，如果不指定在大数据量的情况下文件输出会自动拆分
    df.coalesce(1).write.format("csv").save(Path, mode='overwrite')

def split_data_by_category(df,col_name,path):
    """
    按照产品类别ID进行结果拆分
    :param df:
    :param col_name:
    :param path:
    :return:
    """
    category_id =df.select(col_name).distinct().collect()
    print(category_id)
    for i in category_id:
        v=str(col_name+"="+i.category)
        print("the sql is :".format(v))
        tmp= df.where(v)
        tmp.show(5)
        path_tmp=path+"/"+str(i)
        save_DF(tmp,path_tmp,mode='overwrite')

def test():
    print("the test is ok!")