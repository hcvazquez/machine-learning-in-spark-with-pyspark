# Databricks notebook source
import findspark
findspark.init()

# COMMAND ----------

#import SparkSession
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('log_reg').getOrCreate()

# COMMAND ----------

#read the dataset
df=spark.read.csv('Log_Reg_dataset.csv',inferSchema=True,header=True)

# COMMAND ----------

from pyspark.sql.functions import *


# COMMAND ----------

#check the shape of the data 
print((df.count(),len(df.columns)))

# COMMAND ----------

#printSchema
df.printSchema()

# COMMAND ----------

#number of columns in dataset
df.columns

# COMMAND ----------

#view the dataset
df.show(5)

# COMMAND ----------

#Exploratory Data Analysis
df.describe().show()


# COMMAND ----------

df.groupBy('Country').count().show()

# COMMAND ----------

df = df.withColumnRenamed("Platform", "Search_Engine")
df.groupBy('Search_Engine').count().show()

# COMMAND ----------

df.groupBy('Status').count().show()

# COMMAND ----------

df.groupBy('Country').mean().show()

# COMMAND ----------

df.groupBy('Search_Engine').mean().show()

# COMMAND ----------

df.groupBy('Status').mean().show()

# COMMAND ----------

#converting categorical data to numerical form

# COMMAND ----------

#import required libraries

from pyspark.ml.feature import StringIndexer


# COMMAND ----------

#Indexing 

# COMMAND ----------

search_engine_indexer = StringIndexer(inputCol="Search_Engine", outputCol="Search_Engine_Num").fit(df)
df = search_engine_indexer.transform(df)

# COMMAND ----------

df.show(5,False)

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder

# COMMAND ----------

#one hot encoding
search_engine_encoder = OneHotEncoder(inputCol="Search_Engine_Num", outputCol="Search_Engine_Vector")
df = search_engine_encoder.transform(df)

# COMMAND ----------

df.show(3,False)

# COMMAND ----------

df.groupBy('Search_Engine').count().orderBy('count',ascending=False).show(5,False)

# COMMAND ----------

df.groupBy('Search_Engine_Num').count().orderBy('count',ascending=False).show(5,False)

# COMMAND ----------

df.groupBy('Search_Engine_Vector').count().orderBy('count',ascending=False).show(5,False)

# COMMAND ----------

country_indexer = StringIndexer(inputCol="Country", outputCol="Country_Num").fit(df)
df = country_indexer.transform(df)

# COMMAND ----------

df.select(['Country','Country_Num']).show(3,False)

# COMMAND ----------

#one hot encoding
country_encoder = OneHotEncoder(inputCol="Country_Num", outputCol="Country_Vector")
df = country_encoder.transform(df)

# COMMAND ----------

df.select(['Country','country_Num','Country_Vector']).show(5,False)

# COMMAND ----------

df.groupBy('Country').count().orderBy('count',ascending=False).show(5,False)

# COMMAND ----------

df.groupBy('Country_Num').count().orderBy('count',ascending=False).show(5,False)

# COMMAND ----------

df.groupBy('Country_Vector').count().orderBy('count',ascending=False).show(5,False)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

df_assembler = VectorAssembler(inputCols=['Search_Engine_Vector','Country_Vector','Age', 'Repeat_Visitor','Web_pages_viewed'], outputCol="features")
df = df_assembler.transform(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.select(['features','Status']).show(10,False)

# COMMAND ----------

#select data for building model
model_df=df.select(['features','Status'])

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

#split the data 
training_df,test_df=model_df.randomSplit([0.75,0.25])

# COMMAND ----------

training_df.count()

# COMMAND ----------

training_df.groupBy('Status').count().show()

# COMMAND ----------

test_df.count()

# COMMAND ----------

test_df.groupBy('Status').count().show()

# COMMAND ----------

log_reg=LogisticRegression(labelCol='Status').fit(training_df)

# COMMAND ----------

#Training Results

# COMMAND ----------

train_results=log_reg.evaluate(training_df).predictions

# COMMAND ----------

train_results.filter(train_results['Status']==1).filter(train_results['prediction']==1).select(['Status','prediction','probability']).show(10,False)

# COMMAND ----------

# MAGIC %md Probability at 0 index is for 0 class and probabilty as 1 index is for 1 class

# COMMAND ----------

correct_preds=train_results.filter(train_results['Status']==1).filter(train_results['prediction']==1).count()


# COMMAND ----------

training_df.filter(training_df['Status']==1).count()

# COMMAND ----------

#accuracy on training dataset 
float(correct_preds)/(training_df.filter(training_df['Status']==1).count())

# COMMAND ----------

#Test Set results

# COMMAND ----------

results=log_reg.evaluate(test_df).predictions

# COMMAND ----------

results.select(['Status','prediction']).show(10,False)

# COMMAND ----------

results.printSchema()

# COMMAND ----------



# COMMAND ----------

#confusion matrix
true_postives = results[(results.Status == 1) & (results.prediction == 1)].count()
true_negatives = results[(results.Status == 0) & (results.prediction == 0)].count()
false_positives = results[(results.Status == 0) & (results.prediction == 1)].count()
false_negatives = results[(results.Status == 1) & (results.prediction == 0)].count()

# COMMAND ----------

print (true_postives)
print (true_negatives)
print (false_positives)
print (false_negatives)
print(true_postives+true_negatives+false_positives+false_negatives)
print (results.count())

# COMMAND ----------

recall = float(true_postives)/(true_postives + false_negatives)
print(recall)

# COMMAND ----------

precision = float(true_postives) / (true_postives + false_positives)
print(precision)

# COMMAND ----------

accuracy=float((true_postives+true_negatives) /(results.count()))
print(accuracy)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol='Status')
evaluator.evaluate(results)
results.show(15)

# COMMAND ----------

evaluator.evaluate(results)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='Status')
evaluator.evaluate(results)

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

print(log_reg.explainParams())

# COMMAND ----------


