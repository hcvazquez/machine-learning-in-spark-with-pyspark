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