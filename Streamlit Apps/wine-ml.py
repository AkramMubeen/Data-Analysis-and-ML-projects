import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import optuna

st.write("""
# Wine Class Prediction App

This app predicts the **Wine Class**!
""")
st.write('---')

# Loads the Dataset
wine = datasets.load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
X.rename(columns={'od280/od315_of_diluted_wines': 'od280'}, inplace=True)
Y = pd.DataFrame(wine.target, columns=['alcohol'])
X.drop(columns=['alcohol'], inplace=True)
Y = Y.values.ravel()

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    malic_acid = st.sidebar.slider('malic_acid', float(X.malic_acid.min()), float(X.malic_acid.max()), float(X.malic_acid.mean()))
    ash = st.sidebar.slider('ash', float(X.ash.min()), float(X.ash.max()), float(X.ash.mean()))
    alcalinity_of_ash = st.sidebar.slider('alcalinity_of_ash', float(X.alcalinity_of_ash.min()), float(X.alcalinity_of_ash.max()), float(X.alcalinity_of_ash.mean()))
    magnesium = st.sidebar.slider('magnesium', float(X.magnesium.min()), float(X.magnesium.max()), float(X.magnesium.mean()))
    total_phenols = st.sidebar.slider('total_phenols', float(X.total_phenols.min()), float(X.total_phenols.max()), float(X.total_phenols.mean()))
    flavanoids = st.sidebar.slider('flavanoids', float(X.flavanoids.min()), float(X.flavanoids.max()), float(X.flavanoids.mean()))
    nonflavanoid_phenols = st.sidebar.slider('nonflavanoid_phenols', float(X.nonflavanoid_phenols.min()), float(X.nonflavanoid_phenols.max()), float(X.nonflavanoid_phenols.mean()))
    proanthocyanins = st.sidebar.slider('proanthocyanins', float(X.proanthocyanins.min()), float(X.proanthocyanins.max()), float(X.proanthocyanins.mean()))
    color_intensity = st.sidebar.slider('color_intensity', float(X.color_intensity.min()), float(X.color_intensity.max()), float(X.color_intensity.mean()))
    hue = st.sidebar.slider('hue', float(X.hue.min()), float(X.hue.max()), float(X.hue.mean()))
    od280 = st.sidebar.slider('od280', float(X.od280.min()), float(X.od280.max()), float(X.od280.mean()))
    proline = st.sidebar.slider('proline', float(X.proline.min()), float(X.proline.max()), float(X.proline.mean()))
    data = {'malic_acid': malic_acid,
            'ash': ash,
            'alcalinity_of_ash': alcalinity_of_ash,
            'magnesium': magnesium,
            'total_phenols': total_phenols,
            'flavanoids': flavanoids,
            'nonflavanoid_phenols': nonflavanoid_phenols,
            'proanthocyanins': proanthocyanins,
            'color_intensity': color_intensity,
            'hue': hue,
            'od280': od280,
            'proline': proline}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel
# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')


# Build Regression Model
model = RandomForestClassifier()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of alcohol')
st.write(prediction)
st.write('---')



# Create an optimization button
optimize_button = st.button("Optimize Model")

# Define global variables for optimization
best_params = {}
final_model = None

# Function to optimize the model using Optuna
def optimize_model(X, Y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
        }

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X, Y)
        return -model.score(X, Y)  # Minimize negative accuracy

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    return study.best_params

if optimize_button:
    st.write("Optimizing model...")
    
    best_params = optimize_model(X, Y)
    st.write("Optimization complete!")

    # Train the final model with the best hyperparameters
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X, Y)

    # Apply Model to Make Prediction
    prediction = final_model.predict(df)
    
    st.header('Prediction of alcohol after hyperparameter optimization')
    st.write(prediction)
    st.write('---')

# Show SHAP values only if the model is optimized
if final_model is not None:
    # Explaining the model's predictions using SHAP values
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    
    fig, ax = plt.subplots()
    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    ax = shap.summary_plot(shap_values, X)
    st.pyplot(fig, bbox_inches='tight')
    st.write('---')
    
    fig, ax = plt.subplots()
    plt.title('Feature importance based on SHAP values (Bar)')
    ax = shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(fig, bbox_inches='tight')
