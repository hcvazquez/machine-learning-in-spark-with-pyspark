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
