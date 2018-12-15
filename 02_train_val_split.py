import random
random.seed(1234)
import numpy as np
np.random.seed(3455)

import pandas as pd


def div_mf_ap(cls_count):
    test_gender_ratio = [57, 43]
    test_M_positions_ratio = [50, 50]
    test_F_positions_ratio = [57, 43]

    cls_M = int(cls_count * test_gender_ratio[0] / 100.0)
    cls_F = int(cls_count * test_gender_ratio[1] / 100.0)
    if (cls_M + cls_F) != cls_count:
        cls_F += (cls_count - (cls_M + cls_F))
    assert (cls_M + cls_F) == cls_count

    cls_M_PA = int(cls_M * test_M_positions_ratio[0] / 100.0)
    cls_M_AP = int(cls_M * test_M_positions_ratio[1] / 100.0)
    if (cls_M_PA + cls_M_AP) != cls_M:
        cls_M_AP += (cls_M - (cls_M_PA + cls_M_AP))
    assert (cls_M_PA + cls_M_AP) == cls_M

    cls_F_PA = int(cls_F * test_F_positions_ratio[0] / 100.0)
    cls_F_AP = int(cls_F * test_F_positions_ratio[1] / 100.0)
    if (cls_F_PA + cls_F_AP) != cls_F:
        cls_F_AP += (cls_F - (cls_F_PA + cls_F_AP))
    assert (cls_M_PA + cls_M_AP) == cls_M

    return cls_M_PA, cls_M_AP, cls_F_PA, cls_F_AP


def dev_cls(total_images):
    train_cls = ['Lung Opacity', 'Normal', 'No Lung Opacity / Not Normal']
    train_cls_ratios = [23, 33, 44]
    train_cls_count = [int(float(v)*total_images/100.0) for v in train_cls_ratios]
    assert sum(train_cls_count) == total_images
    pne_cls_count = train_cls_count[0]
    nrm_cls_count = train_cls_count[1]
    nn_cls_count = train_cls_count[2]
    return pne_cls_count, nrm_cls_count, nn_cls_count


def div_bb():
    bb_count = [1, 2, 3, 4]
    bb_ratio = [64.5, 34.4, 1, .1]
    bb_ratio_count = [int(float(v)*pne_cls_count/100.0) for v in bb_ratio]
    bb_ratio_count[-1] += 1
    assert sum(bb_ratio_count) == pne_cls_count
    # return bb_ratio_count[0], bb_ratio_count[1], bb_ratio_count[2], bb_ratio_count[3]
    return bb_ratio_count


def div_age(df, req_cnt):
    test_ages_bin = [3, 7, 12, 17, 22, 27, 32, 37, 42, 47, 51, 56, 61, 66, 71, 76, 81, 86, 91]
    test_ages_count = [7, 8, 27, 49, 57, 72, 59, 68, 64, 123, 141, 125, 108, 57, 23, 8, 3, 1]
    temp_sum = sum(test_ages_count)
    test_ages_count = [int(float(v) * req_cnt / temp_sum) for v in test_ages_count]
    if sum(test_ages_count) > req_cnt:
        assert 0 # Not tested
        a_df = sum(test_ages_count) - req_cnt
        offset_l = 0
        offset_r = -1
        alt_flg = 0
        for id in range(a_df):
            if alt_flg == 0:
                alt_flg = 1
                while offset_l < len(test_ages_count):
                    if test_ages_count[offset_l] == 0:
                        offset_l += 1
                        continue
                    break
                test_ages_count[offset_l] -= 1
            else:
                alt_flg = 0
                while abs(offset_r) < len(test_ages_count):
                    if test_ages_count[offset_r] == 0:
                        offset_r -= 1
                        continue
                    break
                test_ages_count[offset_r] -= 1
    elif sum(test_ages_count) < req_cnt:
        a_df = req_cnt - sum(test_ages_count)
        offset = 0
        for id in range(a_df):
            test_ages_count[9+offset] += 1
            offset += 1
            if offset == 3:
                offset = 0
    assert sum(test_ages_count) == req_cnt

    all_df = pd.DataFrame(columns=df.columns)
    for bin_i, bin_st in enumerate(test_ages_bin[0:-1]):
        req_bin_cnt = test_ages_count[bin_i]
        if req_bin_cnt < 1:
            continue
        bin_en = test_ages_bin[bin_i + 1]
        bin_df = df[(df['PatientAge'] >= bin_st) & (df['PatientAge'] < bin_en)]
        if req_bin_cnt > (bin_df.shape[0] // 2):
            req_bin_cnt = bin_df.shape[0] // 2
        bin_df = bin_df.sample(req_bin_cnt, random_state=0)
        all_df = pd.concat([all_df, bin_df], axis=0)
    if req_cnt != all_df.shape[0]:
        if req_cnt == 1:
            all_df = df.sample(1, random_state=0)
        else:
            a_df = req_cnt - all_df.shape[0]
            temp_df = df.sample(a_df, random_state=0)
            all_df = pd.concat([all_df, temp_df], axis=0)
    assert req_cnt == all_df.shape[0]
    return all_df


def get_img_ids(cls_M_PA, cls_M_AP, cls_F_PA, cls_F_AP, cls_imges):
    df = pd.merge(cls_imges, img_meta_df, on="patientId")
    df_m = df[df['PatientSex'] == 'M']
    df_f = df[df['PatientSex'] == 'F']
    df_m_pa = df_m[df_m['ViewPosition'] == 'PA']
    df_m_ap = df_m[df_m['ViewPosition'] == 'AP']
    df_f_pa = df_f[df_f['ViewPosition'] == 'PA']
    df_f_ap = df_f[df_f['ViewPosition'] == 'AP']

    df_all = div_age(df_m_pa, cls_M_PA)
    df_m_ap = div_age(df_m_ap, cls_M_AP)
    df_f_pa = div_age(df_f_pa, cls_F_PA)
    df_f_ap = div_age(df_f_ap, cls_F_AP)
    df_all = pd.concat([df_all, df_m_ap, df_f_pa, df_f_ap], axis=0)

    return df_all


def cleanup(df):
    max_bb_area = 234700  # 99th percntile
    df = df[df['width'] * df['height'] < max_bb_area]
    return df


assert 0
bbox_path = "../datasets/RSNA_pne/combined_class_info.csv"
img_meta_path = "../datasets/RSNA_pne/image_meta.csv"
bbox_df = pd.read_csv(bbox_path)
img_meta_df = pd.read_csv(img_meta_path)
total_val_images = 1000
pne_cls_count, nrm_cls_count, nn_cls_count = dev_cls(total_val_images)

norm_cls_M_PA, norm_cls_M_AP, norm_cls_F_PA, norm_cls_F_AP = div_mf_ap(nrm_cls_count)
nrm_imges = bbox_df[bbox_df['class'] == 'Normal']
# print(len(nrm_imges))
df_validation = get_img_ids(norm_cls_M_PA, norm_cls_M_AP, norm_cls_F_PA, norm_cls_F_AP, nrm_imges)
nrm_cnt = len(df_validation)

nn_cls_M_PA, nn_cls_M_AP, nn_cls_F_PA, nn_cls_F_AP = div_mf_ap(nn_cls_count)
nn_imges = bbox_df[bbox_df['class'] == 'No Lung Opacity / Not Normal']
# print(len(nn_imges))
tmp_df = get_img_ids(nn_cls_M_PA, nn_cls_M_AP, nn_cls_F_PA, nn_cls_F_AP, nn_imges)
df_validation = pd.concat([df_validation, tmp_df], axis=0)
nn_cnt = len(tmp_df)

pne_imgs_bb = bbox_df[bbox_df['Target'] == 1]
# print(len(np.unique(pne_imgs_bb.patientId.values)))
pne_imgs_bb = cleanup(pne_imgs_bb)
# print(len(np.unique(pne_imgs_bb.patientId.values)))
pne_imgs_bb_grp = pne_imgs_bb.groupby('patientId').size().reset_index(name='boxes')

bb_ratio_count = div_bb()
pnn_cnt = 0
for bb_i, bb in enumerate(bb_ratio_count):
    bb_cls_M_PA, bb_cls_M_AP, bb_cls_F_PA, bb_cls_F_AP = div_mf_ap(bb)
    pne_imgs_bb_i = pne_imgs_bb_grp[pne_imgs_bb_grp['boxes'] == bb_i + 1]
    pne_imgs_bb_i = pne_imgs_bb_i.drop('boxes', 1)
    tmp_df = get_img_ids(bb_cls_M_PA, bb_cls_M_AP, bb_cls_F_PA, bb_cls_F_AP, pne_imgs_bb_i)
    df_validation = pd.concat([df_validation, tmp_df], axis=0)
    pnn_cnt += len(tmp_df)

assert df_validation.shape[0] == total_val_images
# print(df_validation.head())
print(["Valid : ", pnn_cnt + nn_cnt + nrm_cnt, " pne : ", pnn_cnt, " normal : ", nrm_cnt,
       " not normal : ", nn_cnt])

tot_patients = bbox_df['patientId'].nunique()
assert (pne_imgs_bb_grp.shape[0] + nrm_imges.shape[0] + nn_imges.shape[0])

keys = list(df_validation.patientId.values)

train_df_pne = pne_imgs_bb_grp[~pne_imgs_bb_grp['patientId'].str.contains('|'.join(keys))]
train_df_pne = pd.merge(train_df_pne, img_meta_df, on="patientId")
req_norm_sz = (train_df_pne.shape[0] // 2) + 1

train_df_nrm_all = nrm_imges[~nrm_imges['patientId'].str.contains('|'.join(keys))]
norm_cls_M_PA, norm_cls_M_AP, norm_cls_F_PA, norm_cls_F_AP = div_mf_ap(req_norm_sz)
train_df_nrm = get_img_ids(norm_cls_M_PA, norm_cls_M_AP, norm_cls_F_PA, norm_cls_F_AP, train_df_nrm_all)

train_df_nn_all = nn_imges[~nn_imges['patientId'].str.contains('|'.join(keys))]
nn_cls_M_PA, nn_cls_M_AP, nn_cls_F_PA, nn_cls_F_AP = div_mf_ap(req_norm_sz)
train_df_nn = get_img_ids(nn_cls_M_PA, nn_cls_M_AP, nn_cls_F_PA, nn_cls_F_AP, train_df_nn_all)

df_train = pd.concat([train_df_nrm, train_df_nn, train_df_pne], axis=0)
print(["Train : ", df_train.shape[0], " pne : ", train_df_pne.shape[0], " normal : ", train_df_nrm.shape[0],
       " not normal : ", train_df_nn.shape[0]])

# df_train = pd.merge(df_train, img_meta_df, on="patientId")
df_train = df_train.drop('boxes', 1)

# print(df_train.head())
# print(df_train.shape)
# print(df_validation.shape)

#######################################3

fid = open("data/train_1.csv", "w")
fid.write("patientId\n")
for f_name in set(list(df_train.path.values)):
    fid.write(((f_name.replace("\\", "/").split("/"))[-1])[0:-4] + "\n")
fid.close()


fid = open("data/validation.csv", "w")
fid.write("patientId\n")
for f_name in set(list(df_validation.path.values)):
    fid.write(((f_name.replace("\\", "/").split("/"))[-1])[0:-4] + "\n")
fid.close()

keys = list(train_df_nrm.patientId.values)
print([len(train_df_nrm_all), len(set(keys))])
train_df_nrm_all = train_df_nrm_all[~train_df_nrm_all['patientId'].str.contains('|'.join(keys))]
train_df_nrm_pids = list(train_df_nrm_all['patientId'].values)
train_df_nrm_req_len = len(train_df_nrm_pids) // 2

keys = list(train_df_nn.patientId.values)
print([len(train_df_nn_all), len(set(keys))])
train_df_nn_all = train_df_nn_all[~train_df_nn_all['patientId'].str.contains('|'.join(keys))]
print([len(train_df_nn_all)])
train_df_nn_pids = list(train_df_nn_all['patientId'].values)
train_df_nn_req_len = len(train_df_nn_pids) // 2

keys = train_df_nrm_pids[0:train_df_nrm_req_len]
train_df_nrm = train_df_nrm_all[train_df_nrm_all['patientId'].str.contains('|'.join(keys))]
keys = train_df_nn_pids[0:train_df_nn_req_len]
train_df_nn = train_df_nn_all[train_df_nn_all['patientId'].str.contains('|'.join(keys))]
df_train = pd.concat([train_df_nrm, train_df_nn, train_df_pne], axis=0)
print(["Train : ", df_train.shape[0], " pne : ", train_df_pne.shape[0], " normal : ", train_df_nrm.shape[0],
       " not normal : ", train_df_nn.shape[0]])
df_train = df_train.drop('boxes', 1)

fid = open("data/train_2.csv", "w")
fid.write("patientId\n")
for f_name in set(list(df_train.patientId.values)):
    fid.write(f_name + "\n")
fid.close()

keys = train_df_nrm_pids[train_df_nrm_req_len:]
train_df_nrm = train_df_nrm_all[train_df_nrm_all['patientId'].str.contains('|'.join(keys))]
keys = train_df_nn_pids[train_df_nn_req_len:]
train_df_nn = train_df_nn_all[train_df_nn_all['patientId'].str.contains('|'.join(keys))]

df_train = pd.concat([train_df_nrm, train_df_nn, train_df_pne], axis=0)
print(["Train : ", df_train.shape[0], " pne : ", train_df_pne.shape[0], " normal : ", train_df_nrm.shape[0],
       " not normal : ", train_df_nn.shape[0]])
df_train = df_train.drop('boxes', 1)

fid = open("data/train_3.csv", "w")
fid.write("patientId\n")
for f_name in set(list(df_train.patientId.values)):
    fid.write(f_name + "\n")
fid.close()
