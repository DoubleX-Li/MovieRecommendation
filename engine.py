# -*- coding: UTF-8 –*-

import os
from pyspark.mlllib.recommendation import ALS

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_counts_and_averages(ID_and_ratings_tuple):
    """
    计算每部电影的评分总数以及平均评分
    输入一个元组 (movieID, ratings_iterable)
    :return: (movieID, (ratings_count, ratings_avg))
    """
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

class RecommendationEngine:
    """
    电影推荐引擎
    """

    def __count_and_average_ratings(self):
        """
        通过现有的数据self.ratings_RDD更新电影评分总数
        """
        logger.info("电影评分计数中...")
        movie_ID_with_ratings_RDD = self.ratings_RDD.map(lambda x: (x[1], x[2])).groupByKey()
        movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
        self.movies_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

    def __train_model(self):
        """
        使用当前数据集训练ALS模型
        """
        logger.info("训练ALS模型中...")
        self.model = ALS.train(self.ratings_RDD, self.rank, seed=self.seed,
                               iterations=self.iterations, lambda_=self.regularization_parameter)
        logger.info("ALS模型建立成功!")

    def __predict_ratings(self, user_and_movie_RDD):
        """
        当给出格式化的RDD(userID, movieID)时得出预测
        :return: 格式化的RDD(movieTitle, movieRating, numRatings)
        """
        predicted_RDD = self.model.predictAll(user_and_movie_RDD)
        predicted_rating_RDD = predicted_RDD.map(lambda x: (x.product, x.rating))
        predicted_rating_title_and_count_RDD = \
            predicted_rating_RDD.join(self.movies_titles_RDD).join(self.movies_rating_counts_RDD)
        predicted_rating_title_and_count_RDD = \
            predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

        return predicted_rating_title_and_count_RDD

    def add_ratings(self, ratings):
        """
        使用格式(user_id, movie_id, rating)添加一个电影评分（可选）
        """
        # 将评分转换成RDD
        new_ratings_RDD = self.sc.parallelize(ratings)
        # 向现有的ratings_RDD添加新的评分
        self.ratings_RDD = self.ratings_RDD.union(new_ratings_RDD)
        # 重新计算电影评分数
        self.__count_and_average_ratings()
        # 使用新的评分重新训练ALS模型
        self.__train_model()

        return ratings

    def get_ratings_for_movie_ids(self, user_id, movie_ids):
        """
        给出一个用户ID和一个电影ID列表，得出该用户对这些电影的评分预测
        """
        requested_movies_RDD = self.sc.parallelize(movie_ids).map(lambda x: (user_id, x))
        # 获取预测的评分
        ratings = self.__predict_ratings(requested_movies_RDD).collect()

        return ratings

    def get_top_ratings(self, user_id, movies_count):
        """
        对输入ID的用户，推荐以movies_count排序的未评分电影
        """
        # 对用户未评分的电影，获取(userID, movieID)
        user_unrated_movies_rdd = self.ratings_RDD.filter(lambda rating: not rating[0] == user_id)\
                                                 .map(lambda x: (user_id, x[1])).distinct()
        # 获取预测的评分
        ratings = self.__predict_ratings(user_unrated_movies_rdd).filter(lambda r: r[2]>=25).takeOrdered(movies_count, key=lambda x: -x[1])

        return ratings

    def __init__(self, sc, dataset_path):
        """
        使用给出的Spark context和数据集路径初始化推荐引擎
        """

        logger.info("启动推荐引擎: ")

        self.sc = sc

        # 为当前用户载入评分数据
        logger.info("载入评分数据中...")
        ratings_file_path = os.path.join(dataset_path, 'ratings.csv')
        ratings_raw_RDD = self.sc.textFile(ratings_file_path)
        ratings_raw_data_header = ratings_raw_RDD.take(1)[0]
        self.ratings_RDD = ratings_raw_RDD.filter(lambda line: line!=ratings_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
        # 为当前用户载入电影数据
        logger.info("载入电影数据中...")
        movies_file_path = os.path.join(dataset_path, 'movies.csv')
        movies_raw_RDD = self.sc.textFile(movies_file_path)
        movies_raw_data_header = movies_raw_RDD.take(1)[0]
        self.movies_RDD = movies_raw_RDD.filter(lambda line: line!=movies_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
        self.movies_titles_RDD = self.movies_RDD.map(lambda x: (int(x[0]),x[1])).cache()
        # 预先计算电影的评分数
        self.__count_and_average_ratings()

        # 训练模型
        self.rank = 8
        self.seed = 5
        self.iterations = 10
        self.regularization_parameter = 0.1
        self.__train_model()