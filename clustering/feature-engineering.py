# Feature Engineering

from pyspark.ml.feature import VectorAssembler

input_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Transform all features into a vector using VectorAssembler
vec_assembler = VectorAssembler(inputCols = input_cols, outputCol='features')
final_data = vec_assembler.transform(df)