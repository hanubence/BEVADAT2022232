from LinearRegressionSkeleton import LinearRegression

model = LinearRegression(1000, 0.0001)

model.fit(model.X_train, model.y_train)
print(model.predict(model.X_test))


