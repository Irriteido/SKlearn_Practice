import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#
fraud_data = pd.read_csv("Dataset\creditcard_2023.csv")
fraud_data = fraud_data.drop("id",axis=1)

#selecting variables and scaling the data
data_x = fraud_data.drop("Class",axis=1)
scale_x = scale(data_x)
data_y = pd.DataFrame(fraud_data["Class"])
y_class = list(fraud_data["Class"].unique())
train_x, test_x, train_y, test_y = train_test_split(scale_x, data_y, test_size=0.2, random_state= 7)

#model
model_kmeans = KMeans(n_clusters=2, random_state= 7)
model_kmeans.fit(train_x)
pred_y = model_kmeans.predict(test_x)


#checking model accuracy
cm = confusion_matrix(test_y, pred_y)
print(cm)
print(f"F1_Score with weighted average = {f1_score(test_y, pred_y, average="weighted")}")
print(f"Model Accuracy = {accuracy_score(test_y, pred_y)}")


#getting the centroids
centroid_x, centroid_y = model_kmeans.cluster_centers_




"""
need to study plotting



#plotting
plt.scatter(data_x["V7"],data_x["V11"], cmap = "summer")
plt.scatter(centroid_x,centroid_y, c="r")
plt.show()



"""

