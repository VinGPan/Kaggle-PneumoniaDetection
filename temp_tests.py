
# import pandas as pd
# import glob
#
# vals = pd.read_csv("data/validation.csv")
# vals = vals['patientId'].values
#
# val_mgs = glob.glob("../rsna_pne/data/val/images/*.png")
# for v in val_mgs:
#     v = (v.replace("\\", "/").split("/")[-1])[0:-4]
#     assert v in vals
#
# vals = pd.read_csv("data/train_1.csv")
# vals = vals['patientId'].values
#
# val_mgs = glob.glob("../rsna_pne/data/train/images/*.png")
# for v in val_mgs:
#     v = (v.replace("\\", "/").split("/")[-1])[0:-4]
#     assert v in vals

