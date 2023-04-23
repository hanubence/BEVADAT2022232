from LinearRegressionSkeleton import LinearRegression

model = LinearRegression(3000, 0.0005)

model.fit(model.X_train, model.y_train)
print(model.predict(model.X_test))


