from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))
from load_model import xg_model, rf_model

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class UserInput(BaseModel):
    Store : int
    DayOfWeek : int
    Sales : int
    Customers : int
    Open : bool
    Promo : int
    StateHoliday : bool
    SchoolHoliday : bool
    StoreType : str
    Assortment : str
    CompetitionDistance : float
    CompetitionOpenSinceMonth : float
    CompetitionOpenSinceYear : float
    Promo2 : int
    Promo2SinceWeek : float
    Promo2SinceYear : float
    PromoInterval : int

app = FastAPI()




@app.post("/predict")
def predict(input:UserInput):
    input_data = np.array([[
                input.Store,  input.DayOfWeek,  input.Sales, input.Customers, input.Open, input.Promo,
                input.StateHoliday,  input.StateHoliday, input.StoreType, input.Assortment, input.CompetitionDistance,
                input.CompetitionOpenSinceMonth, input.CompetitionOpenSinceYear, input.Promo2, input.Promo2SinceWeek, input.Promo2SinceYear,
                input.PromoInterval
            ]], dtype=object) 
    

    # Convert to DataFrame for easier processing
    columns = [
        "Store", "DayOfWeek", "Sales", "Customers", "Open", "Promo", "StateHoliday", 
        "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenSinceMonth", 
        "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", 
        "PromoInterval"
    ]
    df = pd.DataFrame(input_data, columns=columns)

    # Define column types
    numerical_features = [
        "Sales", "Customers", "CompetitionDistance", "CompetitionOpenSinceMonth", 
        "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"
    ]
    categorical_features = [
        "Store", "DayOfWeek", "Open", "Promo", "StateHoliday", "StoreType", 
        "Assortment", "Promo2", "PromoInterval"
    ]

    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    # Apply transformations
    processed_data = preprocessor.fit_transform(df)

    predicted_result = rf_model.predict(processed_data)
    return predicted_result


