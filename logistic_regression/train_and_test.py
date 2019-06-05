from pyspark.ml.classification import LogisticRegression

#split the data 
training_df,test_df=model_df.randomSplit([0.75,0.25])
training_df.count()
training_df.groupBy('Status').count().show()
test_df.count()
test_df.groupBy('Status').count().show()
log_reg=LogisticRegression(labelCol='Status').fit(training_df)

#Training Results
train_results=log_reg.evaluate(training_df).predictions
train_results.filter(train_results['Status']==1).filter(train_results['prediction']==1).select(['Status','prediction','probability']).show(10,False)

# MAGIC %md Probability at 0 index is for 0 class and probabilty as 1 index is for 1 class
correct_preds=train_results.filter(train_results['Status']==1).filter(train_results['prediction']==1).count()
training_df.filter(training_df['Status']==1).count()

#accuracy on training dataset 
float(correct_preds)/(training_df.filter(training_df['Status']==1).count())

#Test Set results

results=log_reg.evaluate(test_df).predictions
results.select(['Status','prediction']).show(10,False)
results.printSchema()

#confusion matrix
true_postives = results[(results.Status == 1) & (results.prediction == 1)].count()
true_negatives = results[(results.Status == 0) & (results.prediction == 0)].count()
false_positives = results[(results.Status == 0) & (results.prediction == 1)].count()
false_negatives = results[(results.Status == 1) & (results.prediction == 0)].count()

print (true_postives)
print (true_negatives)
print (false_positives)
print (false_negatives)
print(true_postives+true_negatives+false_positives+false_negatives)
print (results.count())

recall = float(true_postives)/(true_postives + false_negatives)
print(recall)

precision = float(true_postives) / (true_postives + false_positives)
print(precision)

accuracy=float((true_postives+true_negatives) /(results.count()))
print(accuracy)