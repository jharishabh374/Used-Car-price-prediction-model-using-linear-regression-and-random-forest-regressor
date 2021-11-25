import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt


def vectorize(column):
    col_name = column.name
    unique_keys = column.unique()
    columns = [col_name + str(unique_keys[i]) for i in range(len(unique_keys))]
    index = {}
    for i in range(len(unique_keys)):
        index[unique_keys[i]] = i

    result = []
    for col_value in column:
        temp = [0]*len(unique_keys)
        temp[index[col_value]] = 1
        result.append(temp)
    return pd.DataFrame(result, columns=columns)


df = pd.read_csv("dataset.csv")
df["age"] = 2021-df["year"]

for column in ["fuel", "transmission", "seller_type", "owner", "Car Name"]:
    temp = vectorize(df[column])
    df = pd.concat([df, temp], axis=1)

df = df.drop(["year", "fuel", "seller_type", "transmission", 'owner', "Car Name"], axis=1)
df= df.sample(frac=1)
# df = df.drop(4147)
y = df["selling_price"]
x = df.drop(["selling_price"], axis=1)

split = int(0.8*len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

model= RandomForestRegressor()
# model1= LR()
# model1.coef_ = [0]*len(model1.coef_)
rr=model.fit(x_train, y_train)
# lr= model1.fit(x_train,y_train)

print("The score i.e. the R square value for RandomForest Regressor is:")
print(model.score(x_test, y_test))

#lr = LR().fit(x_train, y_train)
pred = np.array(rr.predict(x_test))
#error = (pred-y_test)

#
plt.plot([i for i in range(len(pred))], y_test, color='green')
plt.plot([i for i in range(len(pred))], pred, color='red')
plt.show()
