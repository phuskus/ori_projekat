import pandas
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from tabulate import tabulate
"""
PROGRAM PARAMETERS
"""
CSV_PATH = "credit_card_data.csv"
CLUSTER_RANGE = (2, 15)

""" 
Load raw CSV
"""
print("Loading data from", CSV_PATH)
dataframe = pandas.read_csv(CSV_PATH).replace(np.nan, 0.0)
"""
Convert to numpy array and ignore CUSTID
"""
print("Converting to numpy array")
np_array = dataframe.iloc[:, 1:].to_numpy()

"""
Experiment with range of cluster numbers and
find the best one using the elbow method
"""
print("Finding best number of clusters")
model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(2,15))

visualizer.fit(np_array)
cluster_num = visualizer.elbow_value_

"""
Fit data with found number of clusters
"""
print("Fitting data to %d clusters" % cluster_num)
kmeans = KMeans(n_clusters=cluster_num)
kmeans.fit(np_array)

"""
Pretty print cluster center data
"""
headers = ["Group"]
headers.extend(dataframe.columns.values[1:])
table = kmeans.cluster_centers_.tolist()
for i in range(len(table)):
    table[i].insert(0, i+1)

print(tabulate(table, headers=headers))
