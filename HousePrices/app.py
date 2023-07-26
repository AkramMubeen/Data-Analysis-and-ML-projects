from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json
from preprocess import make_df,load_and_apply_label_encoders,load_and_apply_scaler

app = Flask(__name__)

columns_to_scale = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
columns_to_encode = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition','MSSubClass','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','MoSold','YrSold','GaragePresence']

scaler_path = "scaler/scaler.pkl"

file_paths = [
    "label_encoders/label_Alley.pkl", "label_encoders/label_BedroomAbvGr.pkl", "label_encoders/label_BldgType.pkl","label_encoders/label_BsmtCond.pkl", "label_encoders/label_BsmtExposure.pkl", "label_encoders/label_BsmtFinType1.pkl","label_encoders/label_BsmtFinType2.pkl", "label_encoders/label_BsmtFullBath.pkl", "label_encoders/label_BsmtHalfBath.pkl","label_encoders/label_BsmtQual.pkl", "label_encoders/label_CentralAir.pkl", "label_encoders/label_Condition1.pkl","label_encoders/label_Condition2.pkl", "label_encoders/label_Electrical.pkl", "label_encoders/label_ExterCond.pkl","label_encoders/label_Exterior1st.pkl", "label_encoders/label_Exterior2nd.pkl", "label_encoders/label_ExterQual.pkl","label_encoders/label_Fence.pkl", "label_encoders/label_FireplaceQu.pkl", "label_encoders/label_Fireplaces.pkl","label_encoders/label_Foundation.pkl", "label_encoders/label_FullBath.pkl", "label_encoders/label_Functional.pkl","label_encoders/label_GarageCars.pkl", "label_encoders/label_GarageCond.pkl", "label_encoders/label_GarageFinish.pkl","label_encoders/label_GaragePresence.pkl", "label_encoders/label_GarageQual.pkl", "label_encoders/label_GarageType.pkl","label_encoders/label_HalfBath.pkl", "label_encoders/label_Heating.pkl", "label_encoders/label_HeatingQC.pkl","label_encoders/label_HouseStyle.pkl", "label_encoders/label_KitchenAbvGr.pkl", "label_encoders/label_KitchenQual.pkl","label_encoders/label_LandContour.pkl", "label_encoders/label_LandSlope.pkl", "label_encoders/label_LotConfig.pkl","label_encoders/label_LotShape.pkl", "label_encoders/label_MasVnrType.pkl", "label_encoders/label_MiscFeature.pkl","label_encoders/label_MoSold.pkl", "label_encoders/label_MSSubClass.pkl", "label_encoders/label_MSZoning.pkl","label_encoders/label_Neighborhood.pkl", "label_encoders/label_OverallCond.pkl", "label_encoders/label_OverallQual.pkl","label_encoders/label_PavedDrive.pkl", "label_encoders/label_PoolQC.pkl", "label_encoders/label_RoofMatl.pkl","label_encoders/label_RoofStyle.pkl", "label_encoders/label_SaleCondition.pkl", "label_encoders/label_SaleType.pkl","label_encoders/label_Street.pkl", "label_encoders/label_TotRmsAbvGrd.pkl", "label_encoders/label_Utilities.pkl","label_encoders/label_YearBuilt.pkl", "label_encoders/label_YearRemodAdd.pkl", "label_encoders/label_YrSold.pkl"
]


# Load the KMeans models for each cluster
with open("models/KMeans.pkl", 'rb') as file:
    KMeans = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = make_df(data)
    df = load_and_apply_label_encoders(df, columns_to_encode, file_paths)
    df = load_and_apply_scaler(df,columns_to_scale,scaler_path)
    # Predict the cluster for the input data using the pre-trained KMeans models
    cluster_id = KMeans.predict(df)[0]

    prediction,cluster_id = model_predict(cluster_id,df)
    print(prediction,cluster_id)
    cluster_id = int(cluster_id)

    # Prepare the response
    response = {
        'Prediction': prediction.tolist(),
        'Cluster': cluster_id
    }
    return jsonify(response)

def model_predict(cluster_id,df):
    # Load the corresponding model for the predicted cluster
    if cluster_id == 0:
        # Cluster 0
        with open('models/XGBRegressor_0.pkl', 'rb') as file:
            model = pickle.load(file)
    elif cluster_id == 1:
        # Cluster 1
        with open('models/XGBRegressor_1.pkl', 'rb') as file:
            model = pickle.load(file)
    elif cluster_id == 2:
        # Cluster 2
        with open('models/XGBRegressor_2.pkl', 'rb') as file:
            model = pickle.load(file)
    elif cluster_id == 3:    
        # Cluster 3
        with open('models/RandomForestRegressor_3.pkl', 'rb') as file:
            model = pickle.load(file)

    # Make predictions using the selected model
    prediction = model.predict(df)

    return prediction[0],cluster_id

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9696,debug=True)
