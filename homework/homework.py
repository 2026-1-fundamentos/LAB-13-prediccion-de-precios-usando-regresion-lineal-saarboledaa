#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import os
import gzip
import pickle
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
import json


def clean_data(df):
    df = df.copy()
    df["Age"] = 2021 - df["Year"]
    df.drop(columns=["Year", "Car_Name"], inplace=True)
    return df


def add_metrics(dataset, y_true, y_pred):
    metrics.append(
        {
            "type": "metrics",
            "dataset": dataset,
            "r2": r2_score(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "mad": mean_absolute_error(y_true, y_pred),
        }
    )


train = pd.read_csv("files/input/train_data.csv.zip")
test = pd.read_csv("files/input/test_data.csv.zip")

train = clean_data(train)
test = clean_data(test)

x_train = train.drop(columns=["Present_Price"])
y_train = train["Present_Price"]

x_test = test.drop(columns=["Present_Price"])
y_test = test["Present_Price"]

categoricas = ["Fuel_Type", "Selling_type", "Transmission"]
numericas = ["Selling_Price", "Driven_kms", "Age", "Owner"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("num", MinMaxScaler(), numericas), 
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(score_func=f_regression)),
        ("regressor", LinearRegression())
    ]
)

param_grid = {
    "selector__k": range(1, 14),
}

model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=10,
    n_jobs=-1,
    refit=True,
)

model.fit(x_train, y_train)

os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(model, f)

os.makedirs("files/output", exist_ok=True)

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

metrics = []

add_metrics("train", y_train, train_pred)
add_metrics("test", y_test, test_pred)

with open("files/output/metrics.json", "w") as f:

    for item in metrics:
        json.dump(item, f)
        f.write("\n")
