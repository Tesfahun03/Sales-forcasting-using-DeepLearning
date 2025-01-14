import os
import sys
import joblib
import numpy as np
sys.path.append(os.path.abspath('..'))
model1 = '../models/2025-01-14-12-08-16.pkl'
model2 = '../models/2025-01-14-12-24-41.pkl'
# loding our model
xg_model = joblib.load(model1)
rf_model = joblib.load(model2)
