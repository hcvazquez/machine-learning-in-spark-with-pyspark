# Databricks notebook source
import findspark
findspark.init()

# COMMAND ----------

#import SparkSession
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('random_forest').getOrCreate()

# COMMAND ----------

#read the dataset
df=spark.read.csv('affairs.csv',inferSchema=True,header=True)

# COMMAND ----------

#check the shape of the data 
print((df.count(),len(df.columns)))

# COMMAND ----------

#printSchema
df.printSchema()

# COMMAND ----------

#view the dataset
df.show(5)

# COMMAND ----------

#Exploratory Data Analysis
df.describe().select('summary','rate_marriage','age','yrs_married','children','religious').show()

# COMMAND ----------

df.groupBy('affairs').count().show()

# COMMAND ----------

df.groupBy('rate_marriage').count().show()

# COMMAND ----------

df.groupBy('rate_marriage','affairs').count().orderBy('rate_marriage','affairs','count',ascending=True).show()

# COMMAND ----------

df.groupBy('religious','affairs').count().orderBy('religious','affairs','count',ascending=True).show()

# COMMAND ----------

df.groupBy('children','affairs').count().orderBy('children','affairs','count',ascending=True).show()

# COMMAND ----------

df.groupBy('affairs').mean().show(5,True)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

df_assembler = VectorAssembler(inputCols=['rate_marriage', 'age', 'yrs_married', 'children', 'religious'], outputCol="features")
df = df_assembler.transform(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.select(['features','affairs']).show(10,False)

# COMMAND ----------

#select data for building model
model_df=df.select(['features','affairs'])

# COMMAND ----------

train_df,test_df=model_df.randomSplit([0.75,0.25])

# COMMAND ----------

train_df.count()

# COMMAND ----------

train_df.groupBy('affairs').count().show()

# COMMAND ----------

test_df.groupBy('affairs').count().show()

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

rf_classifier=RandomForestClassifier(labelCol='affairs',numTrees=50).fit(train_df)

# COMMAND ----------

rf_predictions=rf_classifier.transform(test_df)

# COMMAND ----------

rf_predictions.show()

# COMMAND ----------

rf_predictions.groupBy('prediction').count().show()

# COMMAND ----------

rf_predictions.select(['probability','affairs','prediction']).show(10,False)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

rf_accuracy=MulticlassClassificationEvaluator(labelCol='affairs',metricName='accuracy').evaluate(rf_predictions)

# COMMAND ----------

print('The accuracy of RF on test data is {0:.0%}'.format(rf_accuracy))

# COMMAND ----------

print(rf_accuracy)

# COMMAND ----------

rf_precision=MulticlassClassificationEvaluator(labelCol='affairs',metricName='weightedPrecision').evaluate(rf_predictions)

# COMMAND ----------

print('The precision rate on test data is {0:.0%}'.format(rf_precision))

# COMMAND ----------

rf_precision

# COMMAND ----------

rf_auc=BinaryClassificationEvaluator(labelCol='affairs').evaluate(rf_predictions)

# COMMAND ----------

print(rf_auc)

# COMMAND ----------

# Feature importance

# COMMAND ----------

rf_classifier.featureImportances

# COMMAND ----------

df.schema["features"].metadata["ml_attr"]["attrs"]

# COMMAND ----------

# Save the model 

# COMMAND ----------

pwd

# COMMAND ----------

rf_classifier.save("C:\\Users\\Hernan\\Data Science\\SPARK\\machine-learning-with-pyspark\\chapter_6_Random_Forests\\RF_model")

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassificationModel

# COMMAND ----------

rf=RandomForestClassificationModel.load("C:\\Users\\Hernan\\Data Science\\SPARK\\machine-learning-with-pyspark\\chapter_6_Random_Forests\\RF_model")

# COMMAND ----------

test_df.show(5)

# COMMAND ----------

model_preditions=rf.transform(test_df)

# COMMAND ----------

model_preditions.show()

# COMMAND ----------

single_df = spark.createDataFrame([[5.0,33.0,5.0,1.0,5.0,0.0]], ['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'affairs'])
single_df = df_assembler.transform(single_df)
single_df = single_df.select(['features','affairs'])

# COMMAND ----------

model_predition=rf.transform(single_df)
model_predition.show()

# COMMAND ----------


