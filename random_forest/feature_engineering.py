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