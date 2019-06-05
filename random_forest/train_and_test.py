from pyspark.ml.classification import RandomForestClassifier

#select data for building model
model_df=df.select(['features','affairs'])
train_df,test_df=model_df.randomSplit([0.75,0.25])
train_df.count()
train_df.groupBy('affairs').count().show()
test_df.groupBy('affairs').count().show()

rf_classifier=RandomForestClassifier(labelCol='affairs',numTrees=50).fit(train_df)
rf_predictions=rf_classifier.transform(test_df)
rf_predictions.show()
rf_predictions.groupBy('prediction').count().show()
rf_predictions.select(['probability','affairs','prediction']).show(10,False)