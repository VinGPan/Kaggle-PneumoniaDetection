import glob
import FCN8
import FCN32
import VGGUnet
import FCN8_aug
import FCN32_aug
import FCN8_mod
import VGGUnet_mod
import mask_rcnn.mrcnn.model as modellib
from mrnn_trainer import InferenceConfig


if __name__ == '__main__':
    # path = "weights/fcn8/finalized_models/*.hdf5"
    # weights = glob.glob(path)
    # for w_idx, w in enumerate(weights):
    #     m = FCN8.FCN8(2, input_height=224, input_width=224)
    #     m.load_weights(w)
    #     m.save("models/f8_" + str(w_idx) + ".h5")
    #
    # path = "weights/fcn32/finalized_models/*.hdf5"
    # weights = glob.glob(path)
    # for w_idx, w in enumerate(weights):
    #     m = FCN32.FCN32(2, input_height=224, input_width=224)
    #     m.load_weights(w)
    #     m.save("models/f3_" + str(w_idx) + ".h5")
    #
    # path = "weights/vgg_unet/finalized_models/*.hdf5"
    # weights = glob.glob(path)
    # for w_idx, w in enumerate(weights):
    #     m = VGGUnet.VGGUnet(2, input_height=224, input_width=224)
    #     m.load_weights(w)
    #     m.save("models/un_" + str(w_idx) + ".h5")

    # path = "weights/fcn8_aug/finalized_models/*.hdf5"
    # weights = glob.glob(path)
    # for w_idx, w in enumerate(weights):
    #     m = FCN8_aug.FCN8_aug(2, input_height=224, input_width=224)
    #     m.load_weights(w)
    #     m.save("models/fa_" + str(w_idx) + ".h5")

    # path = "weights/fcn32_aug/finalized_models/*.hdf5"
    # weights = glob.glob(path)
    # for w_idx, w in enumerate(weights):
    #     m = FCN32_aug.FCN32_aug(2, input_height=224, input_width=224)
    #     m.load_weights(w)
    #     m.save("models/ta_" + str(w_idx) + ".h5")

    # path = "weights/fcn8_mod/finalized_models/*.hdf5"
    # weights = glob.glob(path)
    # for w_idx, w in enumerate(weights):
    #     print(w)
    #     m = FCN8_mod.FCN8_mod(2, input_height=224, input_width=224)
    #     m.load_weights(w)
    #     m.save("models/fm_" + str(w_idx) + ".h5")

    path = "weights/vggunet_mod/finalized_models/*.hdf5"
    weights = glob.glob(path)
    for w_idx, w in enumerate(weights):
        print(w)
        m = VGGUnet_mod.VGGUnet_mod(2, input_height=224, input_width=224)
        m.load_weights(w)
        m.save("models/um_" + str(w_idx) + ".h5")

    # path = "weights/mrnn/finalized_models/*.h5"
    # MODEL_DIR = "weights/mrnn/finalized_models/"
    # weights = glob.glob(path)
    # for w_idx, w in enumerate(weights):
    #     inference_config = InferenceConfig()
    #     # Recreate the model in inference mode
    #     m = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
    #     m.load_weights(w, by_name=True)
    #     m.save("models/mr_" + str(w_idx) + ".h5")
