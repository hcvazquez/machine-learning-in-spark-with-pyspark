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

#Exploratory Data Analysis
df.describe().select('summary','rate_marriage','age','yrs_married','children','religious').show()
df.groupBy('affairs').count().show()
df.groupBy('rate_marriage').count().show()
df.groupBy('rate_marriage','affairs').count().orderBy('rate_marriage','affairs','count',ascending=True).show()
df.groupBy('religious','affairs').count().orderBy('religious','affairs','count',ascending=True).show()
df.groupBy('children','affairs').count().orderBy('children','affairs','count',ascending=True).show()
df.groupBy('affairs').mean().show(5,True)


from pyspark.ml.feature import VectorAssembler

df_assembler = VectorAssembler(inputCols=['rate_marriage', 'age', 'yrs_married', 'children', 'religious'], outputCol="features")
df = df_assembler.transform(df)
df.printSchema()
df.select(['features','affairs']).show(10,False)

#select data for building model
model_df=df.select(['features','affairs'])
train_df,test_df=model_df.randomSplit([0.75,0.25])
train_df.count()
train_df.groupBy('affairs').count().show()
test_df.groupBy('affairs').count().show()

from pyspark.ml.classification import RandomForestClassifier

rf_classifier=RandomForestClassifier(labelCol='affairs',numTrees=50).fit(train_df)
rf_predictions=rf_classifier.transform(test_df)
rf_predictions.show()
rf_predictions.groupBy('prediction').count().show()
rf_predictions.select(['probability','affairs','prediction']).show(10,False)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

rf_accuracy=MulticlassClassificationEvaluator(labelCol='affairs',metricName='accuracy').evaluate(rf_predictions)
print('The accuracy of RF on test data is {0:.0%}'.format(rf_accuracy))
print(rf_accuracy)

rf_precision=MulticlassClassificationEvaluator(labelCol='affairs',metricName='weightedPrecision').evaluate(rf_predictions)
print('The precision rate on test data is {0:.0%}'.format(rf_precision))

rf_precision

rf_auc=BinaryClassificationEvaluator(labelCol='affairs').evaluate(rf_predictions)
print(rf_auc)

# Feature importance
rf_classifier.featureImportances
df.schema["features"].metadata["ml_attr"]["attrs"]

# Save the model 
rf_classifier.save("C:\\Users\\Hernan\\Data Science\\SPARK\\machine-learning-with-pyspark\\chapter_6_Random_Forests\\RF_model")

from pyspark.ml.classification import RandomForestClassificationModel

rf=RandomForestClassificationModel.load("C:\\Users\\Hernan\\Data Science\\SPARK\\machine-learning-with-pyspark\\chapter_6_Random_Forests\\RF_model")
test_df.show(5)
model_preditions=rf.transform(test_df)
model_preditions.show()

single_df = spark.createDataFrame([[5.0,33.0,5.0,1.0,5.0,0.0]], ['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'affairs'])
single_df = df_assembler.transform(single_df)
single_df = single_df.select(['features','affairs'])

model_predition=rf.transform(single_df)
model_predition.show()
