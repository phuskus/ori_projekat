import pandas
import numpy as np
import os
from shutil import copyfile

def sortImages(csvPath, imgFolderPath, outputFolderPath):
    csv = pandas.read_csv(csvPath)
    csv = csv.replace(np.nan, "", regex=True)
    directory = imgFolderPath
    path = ""
    label = ""
    labels = {}
    print("Sorting images into subfolders...")
    listDir = os.listdir(directory)
    progressStep = int(0.1 * len(listDir))
    if not os.path.exists(outputFolderPath):
        os.mkdir(outputFolderPath)

    missingCount = 0
    for idx, filename in enumerate(listDir):
        if idx % progressStep == 0:
            print(str(int(((idx+1) * 100.0) / len(listDir))) + "% complete")
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue
        try:
            row = csv.loc[csv["X_ray_image_name"] == filename].iloc[0]
        except:
            #print(filename, "not found in csv")
            missingCount += 1
            continue
        label = row['Label'] + row['Label_1_Virus_category']

        if label in labels.keys():
            labels[label] += 1
        else:
            labels[label] = 1

        dirPath = os.path.join(outputFolderPath, label)
        if not os.path.exists(dirPath):
            os.mkdir(dirPath)

        copyfile(path, os.path.join(dirPath, filename))

    print("Sorting complete!")
    print("%d filenames were not found in the CSV" % missingCount)
    print("Images copied into subdirectories:")
    print(labels)
    return len(labels)