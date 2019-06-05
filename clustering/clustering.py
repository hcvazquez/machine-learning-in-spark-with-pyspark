# Feature Engineering

from pyspark.ml.feature import VectorAssembler

input_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Transform all features into a vector using VectorAssembler
vec_assembler = VectorAssembler(inputCols = input_cols, outputCol='features')
final_data = vec_assembler.transform(df)


# Hyper parameter tuning

errors=[]

for k in range(2,10):
    kmeans = KMeans(featuresCol='features',k=k)
    model = kmeans.fit(final_data)
    intra_distance = model.computeCost(final_data)
    errors.append(intra_distance)
    print("With K={}".format(k))
    print("Within Set Sum of Squared Errors = " + str(intra_distance))
    print('--'*30)

cluster_number = range(2,10)
plt.scatter(cluster_number,errors)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('SSE')
plt.show()


# Selecting k =3 for kmeans clustering
# Train and Predict

kmeans = KMeans(featuresCol='features',k=3,)
model = kmeans.fit(final_data)
model.transform(final_data).groupBy('prediction').count().show()

predictions=model.transform(final_data)
predictions.columns

predictions.groupBy('species','prediction').count().show()

pandas_df = predictions.toPandas()
pandas_df.sample(5)

import matplotlib.pyplot as plt

cluster_vis = plt.figure(figsize=(15,10)).gca(projection='3d')
cluster_vis.scatter(pandas_df.sepal_length, pandas_df.sepal_width, pandas_df.petal_length, c=pandas_df.prediction,depthshade=False)
plt.show()
