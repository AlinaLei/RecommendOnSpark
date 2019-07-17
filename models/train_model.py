from pyspark.sql import Row
from pyspark.sql import HiveContext
from hive import use_hive
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

def train_model_feature(train_data_path, category_path):
    from config import config
    from data_feature import data_handle
    from models import model_feature

    sc = config.CreateSparkContext()
    print("start test1")
    raw_ratings_rdd = data_handle.read_file_to_RDD(sc,train_data_path,pathtype='local')
    print("start read data test 2")
    ratings_rdd = model_feature.handle_read_data(raw_ratings_rdd, 3)
    try:
        print("start handle data test 3")
        ratings_datas = model_feature.create_als_data(ratings_rdd)
        print("start split data test 4")
        training_ratings, testing_ratings = model_feature.split_train_test_data(ratings_datas)
        print("start train model test 5")
        model, rmse_value, rank, iterations, lambda_ = model_feature.train_model_evaluate(training_ratings, testing_ratings, 10, 4, 0.0001)
        try:
            # save_model(sc, model, "file:///data/lin/savemodel/als_model_test")
            print("start recommned result")
            recommend = model_feature.Recommend(ALS_model=model)
            print("start load category data test 7")
            category = data_handle.read_file_to_RDD(sc, category_path, pathtype='local')
            print("start train model test 6")

            recommendation_all = recommend.map(hadle_result).toDF()
            print("recommendation data is :")
            recommendation_all.show(4)
            print("start train model test 8")
            catrgory_rdd = model_feature.handle_read_data(category, 3, sep=',')
            print("start train model test 9")
            category_df = data_handle.transform_rdd_to_DF(catrgory_rdd, ['products', 'category', 'channel'])
            print(category_df.show(5))
            print("start train model test 10")
            #result = data_handle.handle_DataFrame(recommendation_all, category_df, 'products')
            result= recommendation_all.join(category_df,['products'],"left")
            print("the result type is :{}".format(type(result)))
            result.show(4)
            print("start train model test 11")
            try:
                print("start save result to hive!")
                #data_handle.save_DF(result, "/data/lin/predict_data/recommend_movie_result/test/category_result1")
                hive_context = use_hive.HiveOperate(sc)
                hive_context.df_insert_to_hive(result)

            except Exception as e:
                print(str(e))
                print("save result failed")
            print("start split data test13")
            try:
                data_handle.split_data_by_category(result, 'category', "/data/lin/predict_data/recommend_movie_result/test")
            except Exception as e:
                print(str(e))
                print("split data failed")

        except Exception as e:
            print(str(e))
            print("save model failed")
    except Exception as e:
        print(str(e))

def load_model_feature():
    from config import config
    from data_feature import data_handle
    from models import model_feature
    # 加载模型
    sc = config.CreateSparkContext()
    try:
        recommend = model_feature.LoadModel(sc, "file:///data/lin/savemodel/als_model_test")
        recommendation_all = recommend.map(data_handle.hadle_result).toDF()
        print("the recommendation is :{}".format(recommendation_all.head(3)))
        category = data_handle.read_file_to_RDD(sc, "/data/lin/train_data/user_data/category.txt")
        catrgory_rdd = model_feature.handle_read_data(category, 3, sep=',')
        category_df = data_handle.transform_rdd_to_DF(catrgory_rdd, ['products', 'category', 'channel'])
        result = data_handle.handle_DataFrame(recommendation_all, category_df, 'products')
        data_handle.save_DF(result, "file:///data/lin/predict_data/recommend_movie_result/test")
    except Exception as e:
        print(str(e))
        print("recommend failed")


