import pandas as pd
import numpy as np
from PIL import Image


if __name__ == '__main__':
    df = pd.read_csv("../datasets/RSNA_pne/combined_class_info.csv")
    pr = np.zeros([1024, 1024])
    bbs_cnt = 0.0
    for index, row in df.iterrows():
        if pd.isnull(row['x']):
            continue
        x = int(round(row['x']))
        y = int(round(row['y']))
        w = int(round(row['width']))
        h = int(round(row['height']))
        pr[y:y + h, x:x + w] += 1
        bbs_cnt += 1.0
    pr = pr / bbs_cnt
    pr = pr * 255.0
    pr = Image.fromarray(np.uint8(pr))
    pr.save("data/bb_priors.png")


