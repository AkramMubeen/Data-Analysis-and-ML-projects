# HEART FAILURE PREDICTION
The problem here is to identify and classify patients with different conditions whether they will have a heart failure or not.We use this past data for
predicting the future which may help in early diagnose of heart failure or severe heart conditions helping doctors and patients respectfully.

This is machine learning project that uses Heart Failure Prediction dataset from Kaggle datasets.
You can download the dataset directly or use opendatasets library to download it directly.
Here is the link to it: `https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction`
Remember to look at the Kaggle API Docs for downloading Kaggle datasets.

You can view the notebook which includes all the working involving data preparation and cleaning , Exploratory Data Analysis ,feature importance and correlation, choosing 
best performing Machine learning model and using GridSearchCV to tune the hyperparameters of the model. The notebook is filled with markdowns so
it will be easy to understand.

You can run `train.ipynb` to use the best performing model and predict the heart failure of a patient. The file `service.py` is already created so you will just need to
run the command `bentoml serve service:svc`. It will open the Swagger UI there you can use the data in train.ipynb to predict for 
yourselfas a test.
