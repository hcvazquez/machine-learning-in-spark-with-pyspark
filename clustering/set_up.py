# Create Spark Session

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('k_means').getOrCreate()

# Importing libraries

import pyspark
import matplotlib.pyplot as plt
from pyspark.sql.functions import * 
from pyspark.sql.types import *
from pyspark.sql.functions import rand, randn
from pyspark.ml.clustering import KMeans

# Prepare Data

df=spark.read.csv('iris_dataset.csv',inferSchema=True,header=True)
print((df.count(),len(df.columns)))

df.columns
df.printSchema()

df.orderBy(rand()).show(10,False)
df.select('species').distinct().count()
df.groupBy('species').count().orderBy('count',ascending=False).show(10,False)