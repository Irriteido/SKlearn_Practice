import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

#
house_price = pd.read_csv("Dataset\housing_price_dataset.csv")

#replace
unique_neighborhood = house_price["Neighborhood"].unique()
new_price = house_price.replace(
{
    "Neighborhood": [unique_neighborhood[0],unique_neighborhood[1],unique_neighborhood[2]]
},
{
    "Neighborhood": [0,1,2]
}
)

#data
data_x = new_price.drop("Price",axis=1)
data_y = new_price["Price"]
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=7)

#model
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(train_x, train_y)
pred_y = model_lasso.predict(test_x)

#accuracy
print("Mean absolute error =", round(mean_absolute_error(test_y,pred_y)))
print("Median absolute error =", round(median_absolute_error(test_y,pred_y)))
print("Explain variance score =", round(explained_variance_score(test_y,pred_y)))
print("R2 score =", round(r2_score(test_y,pred_y)))

#
fig, ax = plt.subplots(2, 1, layout = "constrained")
ax[0].scatter(test_x["SquareFeet"], test_y, alpha = 0.1)
ax[1].scatter(test_x["SquareFeet"], pred_y,c="r", alpha = 0.1)
plt.show()


