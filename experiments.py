fid = open("experiments.bat", "w")
seed = 42
batchsize = 32
dim = 224
epoch = 100
run_code = 1
models = ['vgg_unet', 'fcn32', 'fcn8']
sdims = [224, 256, 232]
for model_i, model in enumerate(models):
    sdim = sdims[model_i]
    for idx in [1, 2, 3]:
        run_code = idx
        experiment = "python experiment.py --seed " + str(seed) + " --model " + model + \
                     " --batchsize " + str(batchsize) + " --runcode " + str(run_code) + \
                     " --trainidx " + str(idx) + " --dim " + str(dim) + \
                     " --sdim " + str(sdim) + " --epoch " + str(epoch) + "\n"
        fid.write(experiment)
fid.write("pause\n")
fid.close()
