import os

TRAIN_PATH = "input/train"
RELABELS_PATH = "relabels.csv"

os.mkdir("{}/{}".format(TRAIN_PATH, "revise"))

with open(RELABELS_PATH) as f:
    for line in f:
        cols = line.split()
        src = "{}/{}/{}.jpg".format(TRAIN_PATH, cols[1], cols[0])
        dst = "{}/{}/{}.jpg".format(TRAIN_PATH, cols[2], cols[0])

        try:
            os.rename(src, dst)

        except FileNotFoundError:
            print("{} not found".format(src))