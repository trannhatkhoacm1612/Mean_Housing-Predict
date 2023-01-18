from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

x = np.random.rand(100,1)
y = 2 + 3*x

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
print('Weight and Bias: ',reg.coef_,reg.intercept_)
print('Error: ',mean_squared_error(y_test,y_pred))

