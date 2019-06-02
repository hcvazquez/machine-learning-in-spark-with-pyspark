# Databricks notebook source
#create sparksession object
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('lin_reg').getOrCreate()

# COMMAND ----------

#import Linear Regression from spark's MLlib
from pyspark.ml.regression import LinearRegression

# COMMAND ----------

#Load the dataset
df=spark.read.csv('/FileStore/tables/Linear_regression_dataset.csv',inferSchema=True,header=True)

# COMMAND ----------

#validate the size of data
print((df.count(), len(df.columns)))

# COMMAND ----------

#explore the data
df.printSchema()

# COMMAND ----------

#view statistical measures of data 
df.describe().show(5,False)

# COMMAND ----------

#sneak into the dataset
df.head(3)

# COMMAND ----------

#import corr function from pyspark functions
from pyspark.sql.functions import corr

# COMMAND ----------

# check for correlation
df.select([corr(x,'output') for x in df.columns]).show()

# COMMAND ----------

#import vectorassembler to create dense vectors
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

#select the columns to create input vector
df.columns

# COMMAND ----------

#create the vector assembler 
vec_assmebler=VectorAssembler(inputCols=['var_1', 'var_2', 'var_3', 'var_4', 'var_5'],outputCol='features')

# COMMAND ----------

#transform the values
features_df=vec_assmebler.transform(df)

# COMMAND ----------

#validate the presence of dense vectors 
features_df.printSchema()

# COMMAND ----------

#view the details of dense vector
features_df.select('features').show(5,False)

# COMMAND ----------

#create data containing input features and output column
model_df=features_df.select('features','output')

# COMMAND ----------

model_df.show(5,False)

# COMMAND ----------

#size of model df
print((model_df.count(), len(model_df.columns)))

# COMMAND ----------

# MAGIC %md ### Split Data - Train & Test sets

# COMMAND ----------

#split the data into 70/30 ratio for train test purpose
train_df,test_df=model_df.randomSplit([0.7,0.3])

# COMMAND ----------

print((train_df.count(), len(train_df.columns)))

# COMMAND ----------

print((test_df.count(), len(test_df.columns)))

# COMMAND ----------

train_df.describe().show()

# COMMAND ----------

# MAGIC %md ## Build Linear Regression Model 

# COMMAND ----------

#Build Linear Regression model 
lin_Reg=LinearRegression(labelCol='output')

# COMMAND ----------

#fit the linear regression model on training data set 
lr_model=lin_Reg.fit(train_df)

# COMMAND ----------

lr_model.intercept

# COMMAND ----------

print(lr_model.coefficients)

# COMMAND ----------

training_predictions=lr_model.evaluate(train_df)

# COMMAND ----------

training_predictions.meanSquaredError

# COMMAND ----------

training_predictions.r2

# COMMAND ----------

#make predictions on test data 
test_results=lr_model.evaluate(test_df)

# COMMAND ----------

#view the residual errors based on predictions 
test_results.residuals.show(10)

# COMMAND ----------

#coefficient of determination value for model
test_results.r2

# COMMAND ----------

test_results.rootMeanSquaredError

# COMMAND ----------

test_results.meanSquaredError

# COMMAND ----------


