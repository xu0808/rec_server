package com.sparrowrecsys.offline.spark.debug

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, sql}

object ALS {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("featureEngineering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    println("Raw Rating Samples:")
    ratingSamples.printSchema()
    ratingSamples.show(10)
    val training = ratingSamples.select(col("userId").cast(sql.types.IntegerType).as("userIdInt"),
      col("movieId").cast(sql.types.IntegerType).as("movieIdInt"),
      col("rating").cast(sql.types.FloatType).as("ratingFloat"))
    println("Raw training Samples:")
    training.printSchema()
    training.show(10)


    // 建立矩阵分解模型
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userIdInt")
      .setItemCol("movieIdInt")
      .setRatingCol("ratingFloat")


    //训练模型
    val model = als.fit(training)


    //得到物品向量和用户向量
    model.itemFactors.show(10, truncate = false)
    model.userFactors.show(10, truncate = false)
  }
}
