import pandas
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from tabulate import tabulate
import math
"""
PROGRAM PARAMETERS
"""
# Where to load the csv data from
CSV_PATH = "credit_card_data.csv"

# The range of cluster numbers to try out when
# finding the best one
CLUSTER_RANGE = (2, 15)

# Quality levels by which to rank each column of a cluster
# Assumed to be in ascending order
RANKING_CATEGORIES = ["Very low", "Low", "Average", "High", "Very High"]

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
Output group info table
"""
headers = ["Group"]
headers.extend(dataframe.columns.values[1:])
table = kmeans.cluster_centers_.tolist()
for i in range(len(table)):
    table[i].insert(0, i+1)

print(tabulate(table, headers=headers))

groupSizes = [0 for i in range(cluster_num)]
totalSize = len(kmeans.labels_)
for label in kmeans.labels_:
    groupSizes[label] += 1

"""
Output data interpretation
"""
sortedColumns = {}
for i in range(1, len(headers)):
    # Sort groups by this column in ascending order
    col = headers[i]
    sortedColumns[col] = { "min": float("inf"), "max": float("-inf"), "values": [] }
    for row in table:
        if row[i] < sortedColumns[col]["min"]:
            sortedColumns[col]["min"] = row[i]
        if row[i] > sortedColumns[col]["max"]:
            sortedColumns[col]["max"] = row[i]


        if len(sortedColumns[col]) == 0:
            sortedColumns[col]["values"].append(row[i])
            continue

        place = 0
        while place < len(sortedColumns[col]["values"]) and sortedColumns[col]["values"][place] < row[i]:
            place += 1

        if place >= len(sortedColumns[col]):
            sortedColumns[col]["values"].append(row[i])
        else:
            sortedColumns[col]["values"].insert(place, row[i])

rankHeight = 1.0 / len(RANKING_CATEGORIES)
rankings = []
for row in table:
    # Give each column a rank
    myRanking = []
    for i in range(1, len(row)):
        diff = sortedColumns[headers[i]]["max"] - sortedColumns[headers[i]]["min"]
        myHeight = (row[i] - sortedColumns[headers[i]]["min"]) / diff
        index = min(len(RANKING_CATEGORIES)-1, int(myHeight / rankHeight))
        myRanking.append(RANKING_CATEGORIES[index])
    rankings.append(myRanking)

def getLoadBar(rankName):
    bar = ""
    been = False
    for rank in RANKING_CATEGORIES:
        if rank == rankName:
            bar += ">"
            been = True
        elif been:
            bar += " "
        else:
            bar += "="
    return "[%s]" % bar

for i, rankList in enumerate(rankings):
    print("\n\n-----Group %d-----" % (i+1))
    print("Size: %d (%s%%)" % (groupSizes[i], str(round(float(groupSizes[i]) / totalSize * 100.0, 2))))
    table = []
    for j in range(len(rankList)):
        table.append([ headers[j+1], rankList[j], getLoadBar(rankList[j])])
    print(tabulate(table, tablefmt="plain"))