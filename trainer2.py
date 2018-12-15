from loss import Loss
import VGGUnet
import VGGSegnet
import FCN8
import FCN32
from keras.callbacks import ModelCheckpoint
import os
from keras.callbacks import EarlyStopping


class Trainer2:
    def __init__(self, G1, G2, loss, model_name, pre_weights, dim, epochs, train_steps, val_steps, run_code):
        self.G1 = G1.generate
        self.G2 = G2.generate
        self.loss = loss.loss
        self.model_name = model_name
        self.pre_weights = pre_weights
        self.dim = dim
        self.epochs = epochs
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.run_code = run_code

    def train(self):
        modelFns = {'vgg_segnet': VGGSegnet.VGGSegnet, 'vgg_unet': VGGUnet.VGGUnet, 'vgg_unet2': VGGUnet.VGGUnet2,
                    'fcn8': FCN8.FCN8, 'fcn32': FCN32.FCN32}
        modelFN = modelFns[self.model_name]

        m = modelFN(2, input_height=self.dim, input_width=self.dim)
        if self.pre_weights is not None:
            m.load_weights(self.pre_weights)
        m.summary()
        m.compile(loss=self.loss, optimizer="adadelta", metrics=['accuracy', Loss.iou])

        save_weights_path = "weights/" + self.model_name + "/models/"
        if not os.path.exists("weights/" + self.model_name):
            os.makedirs("weights/" + self.model_name)
        if not os.path.exists(save_weights_path):
            os.makedirs(save_weights_path)

        model_path = save_weights_path + str(self.run_code) + ".hdf5"
        modelCheckpoint = ModelCheckpoint(model_path, monitor='val_iou', save_best_only=True, mode='max')
        es = EarlyStopping(monitor='val_iou', patience=5, verbose=0, mode="max")
        m.fit_generator(self.G1(), self.train_steps, epochs=self.epochs,
                        validation_data=self.G2(), validation_steps=self.val_steps,
                        callbacks=[modelCheckpoint, es])


