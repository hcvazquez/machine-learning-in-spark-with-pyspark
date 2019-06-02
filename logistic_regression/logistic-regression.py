# Databricks notebook source
import findspark
findspark.init()


#import SparkSession
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('log_reg').getOrCreate()


#read the dataset
df=spark.read.csv('Log_Reg_dataset.csv',inferSchema=True,header=True)


from pyspark.sql.functions import *



#check the shape of the data 
print((df.count(),len(df.columns)))


#printSchema
df.printSchema()


#number of columns in dataset
df.columns


#view the dataset
df.show(5)


#Exploratory Data Analysis
df.describe().show()



df.groupBy('Country').count().show()


df = df.withColumnRenamed("Platform", "Search_Engine")
df.groupBy('Search_Engine').count().show()


df.groupBy('Status').count().show()


df.groupBy('Country').mean().show()


df.groupBy('Search_Engine').mean().show()


df.groupBy('Status').mean().show()


#converting categorical data to numerical form


#import required libraries

from pyspark.ml.feature import StringIndexer



#Indexing 


search_engine_indexer = StringIndexer(inputCol="Search_Engine", outputCol="Search_Engine_Num").fit(df)
df = search_engine_indexer.transform(df)


df.show(5,False)


from pyspark.ml.feature import OneHotEncoder


#one hot encoding
search_engine_encoder = OneHotEncoder(inputCol="Search_Engine_Num", outputCol="Search_Engine_Vector")
df = search_engine_encoder.transform(df)


df.show(3,False)


df.groupBy('Search_Engine').count().orderBy('count',ascending=False).show(5,False)


df.groupBy('Search_Engine_Num').count().orderBy('count',ascending=False).show(5,False)


df.groupBy('Search_Engine_Vector').count().orderBy('count',ascending=False).show(5,False)


country_indexer = StringIndexer(inputCol="Country", outputCol="Country_Num").fit(df)
df = country_indexer.transform(df)


df.select(['Country','Country_Num']).show(3,False)


#one hot encoding
country_encoder = OneHotEncoder(inputCol="Country_Num", outputCol="Country_Vector")
df = country_encoder.transform(df)


df.select(['Country','country_Num','Country_Vector']).show(5,False)


df.groupBy('Country').count().orderBy('count',ascending=False).show(5,False)


df.groupBy('Country_Num').count().orderBy('count',ascending=False).show(5,False)


df.groupBy('Country_Vector').count().orderBy('count',ascending=False).show(5,False)


from pyspark.ml.feature import VectorAssembler


df_assembler = VectorAssembler(inputCols=['Search_Engine_Vector','Country_Vector','Age', 'Repeat_Visitor','Web_pages_viewed'], outputCol="features")
df = df_assembler.transform(df)


df.printSchema()


df.select(['features','Status']).show(10,False)


#select data for building model
model_df=df.select(['features','Status'])


from pyspark.ml.classification import LogisticRegression


#split the data 
training_df,test_df=model_df.randomSplit([0.75,0.25])


training_df.count()


training_df.groupBy('Status').count().show()


test_df.count()


test_df.groupBy('Status').count().show()


log_reg=LogisticRegression(labelCol='Status').fit(training_df)


#Training Results


train_results=log_reg.evaluate(training_df).predictions


train_results.filter(train_results['Status']==1).filter(train_results['prediction']==1).select(['Status','prediction','probability']).show(10,False)


# MAGIC %md Probability at 0 index is for 0 class and probabilty as 1 index is for 1 class


correct_preds=train_results.filter(train_results['Status']==1).filter(train_results['prediction']==1).count()



training_df.filter(training_df['Status']==1).count()


#accuracy on training dataset 
float(correct_preds)/(training_df.filter(training_df['Status']==1).count())


#Test Set results


results=log_reg.evaluate(test_df).predictions


results.select(['Status','prediction']).show(10,False)


results.printSchema()





#confusion matrix
true_postives = results[(results.Status == 1) & (results.prediction == 1)].count()
true_negatives = results[(results.Status == 0) & (results.prediction == 0)].count()
false_positives = results[(results.Status == 0) & (results.prediction == 1)].count()
false_negatives = results[(results.Status == 1) & (results.prediction == 0)].count()


print (true_postives)
print (true_negatives)
print (false_positives)
print (false_negatives)
print(true_postives+true_negatives+false_positives+false_negatives)
print (results.count())


recall = float(true_postives)/(true_postives + false_negatives)
print(recall)


precision = float(true_postives) / (true_postives + false_positives)
print(precision)


accuracy=float((true_postives+true_negatives) /(results.count()))
print(accuracy)


from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol='Status')
evaluator.evaluate(results)
results.show(15)


evaluator.evaluate(results)


evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='Status')
evaluator.evaluate(results)
evaluator.getMetricName()
print(log_reg.explainParams())