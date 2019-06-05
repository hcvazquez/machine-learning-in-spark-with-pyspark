# Create Spark Session
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('lin_reg').getOrCreate()

#import Linear Regression from spark's MLlib
from pyspark.ml.regression import LinearRegression

#Load the dataset
df=spark.read.csv('/FileStore/tables/Linear_regression_dataset.csv',inferSchema=True,header=True)

#validate the size of data
print((df.count(), len(df.columns)))

#explore the data
df.printSchema()

#view statistical measures of data 
df.describe().show(5,False)

#sneak into the dataset
df.head(3)