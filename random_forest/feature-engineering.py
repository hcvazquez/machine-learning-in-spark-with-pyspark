from pyspark.ml.feature import VectorAssembler

df_assembler = VectorAssembler(inputCols=['rate_marriage', 'age', 'yrs_married', 'children', 'religious'], outputCol="features")
df = df_assembler.transform(df)
df.printSchema()
df.select(['features','affairs']).show(10,False)