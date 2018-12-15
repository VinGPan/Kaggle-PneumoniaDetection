from loss import Loss
import VGGUnet
import VGGSegnet
import FCN8
import FCN32
import FCN8_aug
import FCN32_aug
import FCN8_mod
import VGGUnet_mod


from keras.callbacks import ModelCheckpoint
import os


class Trainer:
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
                    'fcn8': FCN8.FCN8, 'fcn32': FCN32.FCN32, 'fcn8_aug' : FCN8_aug.FCN8_aug,
                    'fcn32_aug': FCN32_aug.FCN32_aug, "fcn8_mod" : FCN8_mod.FCN8_mod,
                    "vggunet_mod": VGGUnet_mod.VGGUnet_mod}
        modelFN = modelFns[self.model_name]

        m = modelFN(2, input_height=self.dim, input_width=self.dim)
        if self.pre_weights is not None:
            m.load_weights(self.pre_weights)
        m.summary()
        m.compile(loss=self.loss, optimizer="adadelta", metrics=['accuracy', Loss.iou])

        save_weights_path = "weights/" + self.model_name + "/" + self.run_code + "/"
        if not os.path.exists("weights/" + self.model_name):
            os.makedirs("weights/" + self.model_name)
        if not os.path.exists(save_weights_path):
            os.makedirs(save_weights_path)

        model_path = save_weights_path + "valid-{epoch:02d}-{acc:.4f}-{loss:.4f}-{iou:.4f}-{val_acc:.4f}-{val_loss:.4f}-{val_iou:.4f}.hdf5"
        model_path2 = save_weights_path + "train-{epoch:02d}-{acc:.4f}-{loss:.4f}-{iou:.4f}-{val_acc:.4f}-{val_loss:.4f}-{val_iou:.4f}.hdf5"
        modelCheckpoint = ModelCheckpoint(model_path, monitor='val_iou', save_best_only=True, mode='max')
        modelCheckpoint2 = ModelCheckpoint(model_path2, monitor='loss')

        m.fit_generator(self.G1(), self.train_steps, epochs=self.epochs,
                        validation_data=self.G2(), validation_steps=self.val_steps,
                        callbacks=[modelCheckpoint, modelCheckpoint2])


if __name__ == '__main__':
    from generator import PreprocImg, Generator
    dim = 224
    sdim = 232
    batch_size = 32
    epochs = 1

    p1 = PreprocImg("resize_divide", [dim])
    p2 = PreprocImg("proc_annot_resize_one_hot", [sdim, 2])

    g1 = Generator("data/train_1.csv", "data/train/images/", batch_size, p1, p2)
    g2 = Generator("data/validation.csv", "data/train/images/", batch_size, p1, p2)

    train_steps = 10000 // batch_size
    val_steps = 1000 // batch_size

    run_code = "1"

    loss = Loss("categor_iou")
    train = Trainer(g1, g2, loss, 'fcn8', None, dim, epochs, train_steps, val_steps, run_code)
    train.train()

