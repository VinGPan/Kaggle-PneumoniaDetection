from mask_rcnn.mrcnn.config import Config
import mask_rcnn.mrcnn.model as modellib
from mask_rcnn.mrcnn import utils
from mask_rcnn.mrcnn import visualize
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class RSNAConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "rsna"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    BATCH_SIZE = IMAGES_PER_GPU
    STEPS_PER_EPOCH = 10000 // BATCH_SIZE

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1000 // BATCH_SIZE

    USE_MINI_MASK = False


class InferenceConfig(RSNAConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class RSNADataset(utils.Dataset):
    def load_shapes(self, imgpath, filenames, dim):
        self.add_class("", 1, "pne")
        self.add_class("", 2, "tmp")
        pid = pd.read_csv(filenames)
        pid = pid['patientId'].values
        imgs = [imgpath + v + ".png" for v in pid]
        for i, path in enumerate(imgs):
            self.add_image("", image_id=i, path=path, dim=dim)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image_name = info['path']
        dim = info['dim']
        img = cv2.imread(image_name, 1)
        # img = cv2.resize(img, (dim, dim))
        return img

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        image_name = info['path'].replace("images", "annotation")
        dim = info['dim']
        import os.path
        if os.path.isfile(image_name):
            img = cv2.imread(image_name, 1)
            # img = cv2.resize(img, (dim, dim))
            img = img[:, :, 0]
        else:
            img = np.zeros((1024, 1024), np.uint8)

        class_ids = []
        # if np.sum(img) > 0:
        class_ids.append(1)
        tmp_img = img * 0
        tmp_img[0:25, 0:25] = 1
        class_ids.append(2)
        class_ids = np.array(class_ids)
        img = np.stack([img, tmp_img], axis=-1)
        # img = np.expand_dims(img, axis=2)
        return img.astype(np.bool), class_ids.astype(np.int32)


class MRNNTrainer:
    def __init__(self, images_path, train_csv_file, test_images_path, test_csv_file):
        self.images_path = images_path
        self.train_csv_file = train_csv_file
        self.test_images_path = test_images_path
        self.test_csv_file = test_csv_file

    def train(self):
        MODEL_DIR = "weights/mrnn/"

        # Local path to trained weights file
        COCO_MODEL_PATH = "../mask_rcnn_coco.h5"

        config = RSNAConfig()
        config.display()

        assert config.IMAGE_SHAPE[0] == config.IMAGE_SHAPE[1]
        dataset_train = RSNADataset()
        dataset_train.load_shapes(self.images_path, self.train_csv_file, config.IMAGE_SHAPE[0])
        dataset_train.prepare()

        # Validation dataset
        dataset_val = RSNADataset()
        dataset_val.load_shapes(self.images_path, "data/validation.csv", config.IMAGE_SHAPE[0])
        dataset_val.prepare()

        # image_ids = np.random.choice(dataset_train.image_ids, 4)
        # for image_id in image_ids:
        #     image = dataset_train.load_image(image_id)
        #     mask, class_ids = dataset_train.load_mask(image_id)
        #     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)

        init_with = "coco"  # imagenet, coco, or last

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last(), by_name=True)

        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=50,
                    layers='heads')

        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=50,
                    layers="all")


    def test(self):
        MODEL_DIR = "weights/mrnn/"
        inference_config = InferenceConfig()

        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir=MODEL_DIR)

        # Get path to saved weights
        # Either set a specific path or find last trained weights
        # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
        model_path = model.find_last()

        # Load trained weights
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        config = RSNAConfig()

        # dataset_train = RSNADataset()
        # dataset_train.load_shapes(self.images_path, self.train_csv_file, config.IMAGE_SHAPE[0])
        # dataset_train.prepare()

        dataset_val = RSNADataset()
        dataset_val.load_shapes(self.test_images_path, "data/test.csv", config.IMAGE_SHAPE[0])
        dataset_val.prepare()

        for image_id in dataset_val.image_ids:
            info = dataset_val.image_info[image_id]
            image_pid = (info['path'].replace("\\", "/").split("/")[-1])[0:-4]

            original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset_val, inference_config,
                                       image_id, use_mini_mask=False)

            # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
            #                             dataset_train.class_names, figsize=(8, 8))

            results = model.detect([original_image], verbose=1)

            r = results[0]

            r['rois'] = r['rois'][r['class_ids'] == 1]
            r['scores'] = r['scores'][r['class_ids'] == 1]
            r['masks'] = r['masks'][:,:,r['class_ids'] == 1]
            r['class_ids'] = r['class_ids'][r['class_ids'] == 1]
            # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
            #                             dataset_val.class_names, r['scores'], figsize=(8, 8))
            # plt.show()

            if r['masks'].shape[2] == 0:
                r['masks'] = np.zeros((256, 256))
            else:
                r['masks'] = r['masks'][:,:] * r['scores']
                r['masks'] = np.sum(r['masks'], axis=-1)
                r['masks'][r['masks'] > 1] = 1
                r['masks'] = r['masks'] * 255

            annot = Image.fromarray(np.uint8(r['masks']))
            annot.save("preds/" + image_pid + ".png")
            h = 0



