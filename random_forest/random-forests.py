from pyspark.ml.classification import RandomForestClassifier

rf_classifier=RandomForestClassifier(labelCol='affairs',numTrees=50).fit(train_df)
rf_predictions=rf_classifier.transform(test_df)
rf_predictions.show()
rf_predictions.groupBy('prediction').count().show()
rf_predictions.select(['probability','affairs','prediction']).show(10,False)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

rf_accuracy=MulticlassClassificationEvaluator(labelCol='affairs',metricName='accuracy').evaluate(rf_predictions)
print('The accuracy of RF on test data is {0:.0%}'.format(rf_accuracy))
print(rf_accuracy)

rf_precision=MulticlassClassificationEvaluator(labelCol='affairs',metricName='weightedPrecision').evaluate(rf_predictions)
print('The precision rate on test data is {0:.0%}'.format(rf_precision))

rf_precision

rf_auc=BinaryClassificationEvaluator(labelCol='affairs').evaluate(rf_predictions)
print(rf_auc)

# Feature importance
rf_classifier.featureImportances
df.schema["features"].metadata["ml_attr"]["attrs"]

# Save the model 
rf_classifier.save("C:\\Users\\Hernan\\Data Science\\SPARK\\machine-learning-with-pyspark\\chapter_6_Random_Forests\\RF_model")

from pyspark.ml.classification import RandomForestClassificationModel

rf=RandomForestClassificationModel.load("C:\\Users\\Hernan\\Data Science\\SPARK\\machine-learning-with-pyspark\\chapter_6_Random_Forests\\RF_model")
test_df.show(5)
model_preditions=rf.transform(test_df)
model_preditions.show()

single_df = spark.createDataFrame([[5.0,33.0,5.0,1.0,5.0,0.0]], ['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'affairs'])
single_df = df_assembler.transform(single_df)
single_df = single_df.select(['features','affairs'])

model_predition=rf.transform(single_df)
model_predition.show()
