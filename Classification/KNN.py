import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#getting dataset and minor preprocessing
sample_data = pd.read_csv("Dataset\cropdamage.csv")

#replacing unique values to numeric
unique_crop = sample_data.iloc[:,2].unique()
unique_soil = sample_data.iloc[:,3].unique()
unique_pesticide = sample_data.iloc[:,4].unique()
unique_season = sample_data.iloc[:,8].unique()

sample_data = sample_data.replace(
{
    "Crop_Type": [unique_crop[0],unique_crop[1]],
    "Soil_Type":[unique_soil[0],unique_soil[1]],
    "Pesticide_Use_Category":[unique_pesticide[0],unique_pesticide[1],unique_pesticide[2]],
    "Season":[unique_season[0],unique_season[1],unique_season[2]]
},
{
    "Crop_Type": [0,1],
    "Soil_Type":[0,1],
    "Pesticide_Use_Category":[0,1,2],
    "Season":[0,1,2]
})

#assigning x and y variables
data_x = sample_data.drop(["ID","Number_Doses_Week","Crop_Damage","CropDamageValue"], axis=1)
data_y = sample_data["CropDamageValue"]
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state= 7)

#scaling data 
scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_x)
scaled_test_x = scaler.transform(test_x)

# no optimization
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(scaled_train_x,train_y)
pred_y = knn.predict(scaled_test_x)

#checking model accuracy
cm = confusion_matrix(test_y, pred_y)
print(cm)
print(f"F1_Score with weighted average = {f1_score(test_y, pred_y, average="weighted")}")
print(f"Model Accuracy = {accuracy_score(test_y, pred_y)}")