import glob
import numpy as np
import concurrent.futures
from functools import partial
from generator import PreprocImg
import VGGUnet
import VGGSegnet
import FCN8
import FCN32
import scipy.misc
import cv2


def approx_to_bb(pr):
    pr = np.array(scipy.misc.toimage(pr))
    last_cnt = 0
    while True:
        _, contours, _ = cv2.findContours(pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bbs = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(pr, (x, y), (x + w, y + h), (255, 255, 255), thickness=cv2.FILLED)
            bbs.append((x, y, w, h))
        if len(contours) == last_cnt:
            break
        last_cnt = len(contours)
    return pr, bbs


def bb_with_conf(pr, pr_bc, tgt_shape):
    pr = cv2.resize(np.array(scipy.misc.toimage(pr)), tgt_shape, interpolation=cv2.INTER_NEAREST)
    pr_bc = cv2.resize(np.array(scipy.misc.toimage(pr_bc)), tgt_shape, interpolation=cv2.INTER_CUBIC)
    pr_bc = pr_bc.astype(np.float32)
    pr_bc = pr_bc / 255.0
    pr, bbs = approx_to_bb(pr)
    bbs_str = ''
    area_th = 10000
    if len(bbs) != 0:
        skip_ids = []
        for idx, bb in enumerate(bbs):
            x, y, w, h = bb
            conf = np.mean(pr_bc[y:y + h, x:x + w])
            if (w * h) < area_th:
                skip_ids.append(idx)
                continue
            if conf <= .5:
                skip_ids.append(idx)
                continue
            bbs[idx] = [conf,
                        str("{:.2f}".format(conf)) + " " + str(int(x)) + " " + str(int(y)) + " " + str(
                            int(w)) + " " + str(
                            int(h))
                        ]
        bbs = [v for i, v in enumerate(bbs) if i not in skip_ids]
        bbs = sorted(bbs, key=lambda x: x[0], reverse=True)
        bbs = bbs[0:3]
        for bb_i, bb in enumerate(bbs):
            bbs_str += (bb[1] + " ")
        bbs_str = bbs_str[0:-1]
        print(bbs_str)
    return bbs_str


def prep_img(dim, img):
    pid = (img.replace("\\", "/").split("/")[-1])[0:-4]
    return [PreprocImg("resize_divide_histeq", [dim]).proc(img), pid]


def prep_data(imgs, dim):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        data = list(executor.map(partial(prep_img, dim), imgs))
    X = np.array([v[0] for v in data])
    pids = [v[1] for v in data]
    return X, pids


if __name__ == '__main__':
    imgs = np.array(sorted(glob.glob("data/test/images/*.png")))
    dim = 224
    sdim = 232
    modelFns = {'vgg_segnet': VGGSegnet.VGGSegnet, 'vgg_unet': VGGUnet.VGGUnet, 'vgg_unet2': VGGUnet.VGGUnet2,
                'fcn8': FCN8.FCN8, 'fcn32': FCN32.FCN32}
    models = []
    model_name = 'fcn8'
    modelFN = modelFns[model_name]
    weights = glob.glob("weights/" + model_name + "/models/*.hdf5")
    for w_i, w in enumerate(weights):
        model = modelFN(2, input_height=dim, input_width=dim)
        model.load_weights(w)
        models.append(model)
    fid = open("submission.csv", "w")
    fid.write("patientId,PredictionString\n")
    pos_pred_cnt = 1
    for i in range(0, len(imgs), 32):
        c_imgs = imgs[i:i+32]
        X, pids = prep_data(c_imgs, dim)
        all_prs = np.zeros((len(c_imgs), sdim * sdim, 2))
        for model in models:
            pr_loc = model.predict(X)
            sm = np.sum(pr_loc, axis=-1)
            pr_loc[:, :, 0] = pr_loc[:, :, 0] / sm
            pr_loc[:, :, 1] = pr_loc[:, :, 1] / sm
            all_prs = all_prs + pr_loc
        all_prs = all_prs / len(models)
        for idx, cpr in enumerate(all_prs):
            pr_bc = cpr.reshape((sdim, sdim, 2))
            pr = pr_bc.argmax(axis=2)
            bbs = bb_with_conf(pr, pr_bc[:, :, 1], (1024, 1024))
            fid.write(pids[idx] + "," + bbs + "\n")
            if bbs != '':
                pos_pred_cnt += 1
    fid.close()
    print(pos_pred_cnt)
