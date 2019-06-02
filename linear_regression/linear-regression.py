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

#import corr function from pyspark functions
from pyspark.sql.functions import corr

# check for correlation
df.select([corr(x,'output') for x in df.columns]).show()

#import vectorassembler to create dense vectors
from pyspark.ml.feature import VectorAssembler

#select the columns to create input vector
df.columns

#create the vector assembler 
vec_assmebler=VectorAssembler(inputCols=['var_1', 'var_2', 'var_3', 'var_4', 'var_5'],outputCol='features')

#transform the values
features_df=vec_assmebler.transform(df)

#validate the presence of dense vectors 
features_df.printSchema()

#view the details of dense vector
features_df.select('features').show(5,False)

#create data containing input features and output column
model_df=features_df.select('features','output')
model_df.show(5,False)

#size of model df
print((model_df.count(), len(model_df.columns)))

# Split Data - Train & Test sets
#split the data into 70/30 ratio for train test purpose
train_df,test_df=model_df.randomSplit([0.7,0.3])

print((train_df.count(), len(train_df.columns)))
print((test_df.count(), len(test_df.columns)))
train_df.describe().show()

# Build Linear Regression Model 
lin_Reg=LinearRegression(labelCol='output')

#fit the linear regression model on training data set 
lr_model=lin_Reg.fit(train_df)
lr_model.intercept
print(lr_model.coefficients)
training_predictions=lr_model.evaluate(train_df)
training_predictions.meanSquaredError
training_predictions.r2

#make predictions on test data 
test_results=lr_model.evaluate(test_df)

#view the residual errors based on predictions 
test_results.residuals.show(10)

#coefficient of determination value for model
test_results.r2
test_results.rootMeanSquaredError
test_results.meanSquaredError
