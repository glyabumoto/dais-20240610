# Databricks notebook source
import pandas as pd
import numpy as np

# test data
np.random.seed(42)
num_points = 100
data = pd.DataFrame({
    'lat': np.random.uniform(low=37.70, high=37.80, size=num_points),
    'lon': np.random.uniform(low=-122.50, high=-122.40, size=num_points),
    'value': np.random.randint(1, 100, size=num_points)
})
data["Name"] = "city"

# make geojson
data_json = data.to_json(orient='records')

print(data_json)


# COMMAND ----------

mapbox_api_key = dbutils.secrets.get(scope="dais", key="mapbox")