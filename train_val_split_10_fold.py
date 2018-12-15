import pandas as pd
from sklearn.cross_validation import KFold
import numpy as np


def get_cls_ratio(pids, bb_df):
    tmp = bb_df[bb_df['patientId'].str.contains('|'.join(pids.tolist()))]
    tmp['area'] = tmp.width * tmp.height
    total_area = 1024 * 1024 * len(pids)
    rt = float(tmp['area'].sum()) / total_area
    return [(1.0 - rt)*100, rt*100]


if __name__ == '__main__':
    bb_df = pd.read_csv("combined_class_info.csv")
    pids = np.array(list(set(bb_df['patientId'].values)))
    nfolds = 10
    random_state = 42
    kf = KFold(len(pids), n_folds=nfolds, shuffle=True, random_state=random_state)
    idx = 1
    fid_rt = open("data/ratios.csv", "w")
    fid_rt.write("set,train_0,train_1,valid_0,valid_1\n")
    for train_index, test_index in kf:
        X_train, X_valid = pids[train_index], pids[test_index]
        fid = open("data/train_" + str(idx) + ".csv", "w")
        fid.write("patientId\n")
        for x in X_train:
            fid.write(x + "\n")
        fid.close()

        fid = open("data/valid_" + str(idx) + ".csv", "w")
        fid.write("patientId\n")
        for x in X_valid:
            fid.write(x + "\n")
        fid.close()

        t = get_cls_ratio(X_train, bb_df)
        v = get_cls_ratio(X_valid, bb_df)
        fid_rt.write(str(idx) + "," + "{:.4f}".format(t[0]) + "," + "{:.4f}".format(t[1]) + "," +
                     "{:.4f}".format(v[0]) + "," + "{:.4f}".format(v[1]) + "\n")
        idx += 1
    fid_rt.close()
