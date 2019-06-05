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