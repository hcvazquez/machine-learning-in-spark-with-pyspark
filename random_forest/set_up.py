# Databricks notebook source
import findspark
findspark.init()

#import SparkSession
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('random_forest').getOrCreate()

#read the dataset
df=spark.read.csv('affairs.csv',inferSchema=True,header=True)

#check the shape of the data 
print((df.count(),len(df.columns)))

#printSchema
df.printSchema()

#view the dataset
df.show(5)