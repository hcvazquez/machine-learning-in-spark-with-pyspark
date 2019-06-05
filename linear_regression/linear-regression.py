# Split Data - Train & Test sets
#split the data into 70/30 ratio for train test purpose
train_df,test_df=model_df.randomSplit([0.7,0.3])

print((train_df.count(), len(train_df.columns)))
print((test_df.count(), len(test_df.columns)))
train_df.describe().show()

# Build Linear Regression Model 
lin_Reg=LinearRegression(labelCol='output')

#fit the linear regression model on training data set 
lr_model=lin_Reg.fit(train_df)
lr_model.intercept
print(lr_model.coefficients)
training_predictions=lr_model.evaluate(train_df)
training_predictions.meanSquaredError
training_predictions.r2

#make predictions on test data 
test_results=lr_model.evaluate(test_df)

#view the residual errors based on predictions 
test_results.residuals.show(10)

#coefficient of determination value for model
test_results.r2
test_results.rootMeanSquaredError
test_results.meanSquaredError
