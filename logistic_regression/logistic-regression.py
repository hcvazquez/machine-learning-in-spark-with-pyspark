from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol='Status')
evaluator.evaluate(results)
results.show(15)
evaluator.evaluate(results)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol='Status')
evaluator.evaluate(results)
evaluator.getMetricName()
print(log_reg.explainParams())