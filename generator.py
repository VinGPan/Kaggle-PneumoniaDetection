import cv2
import numpy as np
import pandas as pd
import itertools


class PreprocImg:
    def __init__(self, typ, args):
        self.proc = None
        self.args = args
        if typ == "resize_divide":
            self.proc = self.proc_resize_divide
        elif typ == "resize_divide_histeq":
            self.proc = self.proc_resize_divide_histeq
        elif typ == "proc_annot_resize_one_hot":
            self.proc = self.proc_annot_resize_one_hot
        else:
            assert 0

    def proc_resize_divide_histeq(self, image_name):
        dim = self.args[0]
        img = cv2.imread(image_name, 0)
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, (dim, dim))
        img = img.astype(np.float32)
        img = img / 255.0
        img = np.stack([img, img, img], axis=-1)
        img = np.rollaxis(img, 2, 0)
        return img

    def proc_resize_divide(self, image_name):
        dim = self.args[0]
        img = cv2.imread(image_name, 1)
        img = cv2.resize(img, (dim, dim))
        img = img.astype(np.float32)
        img = img / 255.0
        img = np.rollaxis(img, 2, 0)
        return img

    def proc_annot_resize_one_hot(self, image_name):
        dim = self.args[0]
        nClasses = self.args[1]
        img = cv2.imread(image_name, 1)
        seg_labels = np.zeros((dim, dim, nClasses))
        img = cv2.resize(img, (dim, dim))
        img = img[:, :, 0]
        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)
        seg_labels = np.reshape(seg_labels, (dim * dim, nClasses))
        return seg_labels


class PreprocSeg:
    def __init__(self):
        pass


class Generator:
    def __init__(self, file_names, path, batch_size, img_proc, annot_proc, aug_data_flg):
        self.batch_size = batch_size
        pid = pd.read_csv(file_names)
        pid = pid['patientId'].values
        aug_df = pd.read_csv("../datasets/RSNA_pne/image_meta.csv")
        imgs = [path + v + ".png" for v in pid]
        annots = [v.replace("images", "annotation") for v in imgs]
        aug = [self.get_aug(v, aug_df) for v in pid]
        self.zipped = itertools.cycle(zip(imgs, annots, aug))
        self.img_proc = img_proc
        self.annot_proc = annot_proc
        self.img_cnt = len(pid)
        self.aug_data_flg = aug_data_flg

    def get_aug(self, v, aug_df):
        tmp = aug_df[aug_df['patientId'] == v]
        age = tmp['PatientAge'].values[0]
        gen = tmp['PatientSex'].values[0]
        if gen == 'M':
            gen = 0
        else:
            gen = 1
        vwp = tmp['ViewPosition'].values[0]
        if vwp == 'AP':
            vwp = 0
        else:
            vwp = 1
        return [age, gen, vwp]

    def generate(self):
        while True:
            X = []
            Y = []
            augs = []
            for _ in range(self.batch_size):
                im, seg, aug = next(self.zipped)
                X.append(self.img_proc.proc(im))
                Y.append(self.annot_proc.proc(seg))
                augs.append(aug)
            if self.aug_data_flg:
                yield [np.array(X), np.array(augs)], np.array(Y)
            else:
                yield np.array(X), np.array(Y)


if __name__ == '__main__':
    p1 = PreprocImg("resize_divide", [224])
    p2 = PreprocImg("proc_annot_resize_one_hot", [232, 2])
    batch_size = 32
    g = Generator("data/validation.csv", "data/train/images/", batch_size, p1, p2)
    g.generate()
