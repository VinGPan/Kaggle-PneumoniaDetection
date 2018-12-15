import random
random.seed(142)
import pandas as pd
import pydicom
import concurrent.futures
from functools import partial
import numpy as np
from PIL import Image
import os
import glob


def make_annots(df1, test_flg, pid):
    src_path = "../datasets/RSNA_pne/stage_1_train_images/" + pid + ".dcm"
    dst_path = "data/train/images/" + pid + ".png"
    if test_flg:
        src_path = "../datasets/RSNA_pne/stage_1_test_images/" + pid + ".dcm"
        dst_path = "data/test/images/" + pid + ".png"
    print([src_path, dst_path])
    img = pydicom.read_file(src_path).pixel_array
    img = Image.fromarray(np.uint8(img))
    img.save(dst_path)

    if test_flg:
        return
    im = Image.open(dst_path)
    annot = np.array(im)
    annot *= 0
    tmp1 = df1[df1['patientId'] == pid]
    for index, row in tmp1.iterrows():
        if pd.isnull(row['x']):
            continue
        x = int(round(row['x']))
        y = int(round(row['y']))
        w = int(round(row['width']))
        h = int(round(row['height']))
        annot[y:y + h, x:x + w] = 1
    annot = Image.fromarray(np.uint8(annot))
    dst_path = dst_path.replace("images", "annotation")
    annot.save(dst_path)


if __name__ == '__main__':
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/train"):
        os.makedirs("data/train")
    if not os.path.exists("data/train/images"):
        os.makedirs("data/train/images")
    if not os.path.exists("data/train/annotation"):
        os.makedirs("data/train/annotation")
    if not os.path.exists("data/test"):
        os.makedirs("data/test")
    if not os.path.exists("data/test/images"):
        os.makedirs("data/test/images")

    df1 = pd.read_csv("../datasets/RSNA_pne/combined_class_info.csv")
    # pids = df1['patientId'].values
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(partial(make_annots, df1, False), pids)

    pids = glob.glob("../datasets/RSNA_pne/stage_1_test_images/*.dcm")
    pids = [((v.replace("\\", "/").strip().split("/"))[-1])[0:-4] for v in pids]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(partial(make_annots, df1, True), pids)

    fid = open("data/test.csv", "w")
    fid.write("patientId\n")
    for p in pids:
        fid.write(p + "\n")
    fid.close()
