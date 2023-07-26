FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app

# Install dependencies
COPY ["Pipfile","Pipfile.lock","./"]
RUN pipenv install --system --deploy

# Copy the app files
COPY ["app.py","preprocess.py","./"]

# Create directories for models, scaler, and label encoders
RUN mkdir -p /app/models
RUN mkdir -p /app/scaler
RUN mkdir -p /app/label_encoders

# Copy model files
COPY ["models/XGBRegressor_2.pkl","models/XGBRegressor_1.pkl","models/XGBRegressor_0.pkl","models/RandomForestRegressor_3.pkl","models/KMeans.pkl", "/app/models/"]

# Copy scaler file
COPY ["scaler/scaler.pkl", "/app/scaler/"]

# Copy label encoder files
COPY ["label_encoders/label_Alley.pkl", "label_encoders/label_BedroomAbvGr.pkl", "label_encoders/label_BldgType.pkl","label_encoders/label_BsmtCond.pkl", "label_encoders/label_BsmtExposure.pkl", "label_encoders/label_BsmtFinType1.pkl","label_encoders/label_BsmtFinType2.pkl", "label_encoders/label_BsmtFullBath.pkl", "label_encoders/label_BsmtHalfBath.pkl","label_encoders/label_BsmtQual.pkl", "label_encoders/label_CentralAir.pkl", "label_encoders/label_Condition1.pkl","label_encoders/label_Condition2.pkl", "label_encoders/label_Electrical.pkl", "label_encoders/label_ExterCond.pkl","label_encoders/label_Exterior1st.pkl", "label_encoders/label_Exterior2nd.pkl", "label_encoders/label_ExterQual.pkl","label_encoders/label_Fence.pkl", "label_encoders/label_FireplaceQu.pkl", "label_encoders/label_Fireplaces.pkl","label_encoders/label_Foundation.pkl", "label_encoders/label_FullBath.pkl", "label_encoders/label_Functional.pkl","label_encoders/label_GarageCars.pkl", "label_encoders/label_GarageCond.pkl", "label_encoders/label_GarageFinish.pkl","label_encoders/label_GaragePresence.pkl", "label_encoders/label_GarageQual.pkl", "label_encoders/label_GarageType.pkl","label_encoders/label_HalfBath.pkl", "label_encoders/label_Heating.pkl", "label_encoders/label_HeatingQC.pkl","label_encoders/label_HouseStyle.pkl", "label_encoders/label_KitchenAbvGr.pkl", "label_encoders/label_KitchenQual.pkl","label_encoders/label_LandContour.pkl", "label_encoders/label_LandSlope.pkl", "label_encoders/label_LotConfig.pkl","label_encoders/label_LotShape.pkl", "label_encoders/label_MasVnrType.pkl", "label_encoders/label_MiscFeature.pkl","label_encoders/label_MoSold.pkl", "label_encoders/label_MSSubClass.pkl", "label_encoders/label_MSZoning.pkl","label_encoders/label_Neighborhood.pkl", "label_encoders/label_OverallCond.pkl", "label_encoders/label_OverallQual.pkl","label_encoders/label_PavedDrive.pkl", "label_encoders/label_PoolQC.pkl", "label_encoders/label_RoofMatl.pkl","label_encoders/label_RoofStyle.pkl", "label_encoders/label_SaleCondition.pkl", "label_encoders/label_SaleType.pkl","label_encoders/label_Street.pkl", "label_encoders/label_TotRmsAbvGrd.pkl", "label_encoders/label_Utilities.pkl","label_encoders/label_YearBuilt.pkl", "label_encoders/label_YearRemodAdd.pkl", "label_encoders/label_YrSold.pkl", "/app/label_encoders/"]

EXPOSE 9696

ENTRYPOINT [ "waitress-serve" ,"--listen=0.0.0.0:9696","app:app" ]
