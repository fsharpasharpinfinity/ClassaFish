import os
import numpy as np

from glob import iglob
from models import Vgg16BN, Inception, Resnet50
from keras.callbacks import ModelCheckpoint

ROOT_DIR = os.getcwd()
DATA_HOME_DIR = ROOT_DIR + '/data'
# %matplotlib inline

# paths
data_path = DATA_HOME_DIR + '/' 
split_train_path = data_path + '/train/'
full_train_path = data_path + '/train_full/'
valid_path = data_path + '/valid/'
test_path = DATA_HOME_DIR + '/test/'
saved_model_path = ROOT_DIR + '/models/'
submission_path = ROOT_DIR + '/submissions/'

# data
batch_size = 16
nb_split_train_samples = 3277
nb_full_train_samples = 3785
nb_valid_samples = 500
nb_test_samples = 1000
classes = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
nb_classes = len(classes)

# model
nb_runs = 5
nb_epoch = 30
nb_aug = 5
dropout = 0.4
clip = 0.01
use_val = True
archs = ["inception"]

models = {
    "vggbn": Vgg16BN(size=(270, 480), n_classes=nb_classes, lr=0.001,
                           batch_size=batch_size, dropout=dropout),
    "inception": Inception(size=(299, 299), n_classes=nb_classes,
                           lr=0.001, batch_size=batch_size),
    "resnet": Resnet50(size=(270, 480), n_classes=nb_classes, lr=0.001,
                    batch_size=batch_size, dropout=dropout)
}

def train(parent_model, model_str):
    parent_model.build()    
    model_fn = saved_model_path + '{val_loss:.2f}-loss_{epoch}epoch_' + model_str
    ckpt = ModelCheckpoint(filepath=model_fn, monitor='val_loss',
                           save_best_only=True, save_weights_only=True)
    
    if use_val:
        parent_model.fit_val(split_train_path, valid_path, nb_trn_samples=nb_split_train_samples, 
                             nb_val_samples=nb_valid_samples, nb_epoch=nb_epoch, callbacks=[ckpt], aug=nb_aug)

        model_path = max(iglob(saved_model_path + '*.h5'), key=os.path.getctime)
        return model_path
    
    model_fn = saved_model_path + '{}epoch_'.format(nb_epoch) + model_str
    parent_model.fit_full(full_train_path, nb_trn_samples=nb_full_train_samples, nb_epoch=nb_epoch, aug=nb_aug)
    model.save_weights(model_fn)
    del parent_model.model 
    
    return model_fn

def train_all():
    	model_paths = {
        "vggbn": [],
        "inception": [],
        'resnet': [],
    }

	for run in range(nb_runs):
		print("Starting Training Run {0} of {1}...\n".format(run+1, nb_runs))
        aug_str = "aug" if nb_aug else "no-aug"
        
        for arch in archs:
        	print("Training {} model...\n".format(arch))
        	model = models[arch]
        	model_str = "{0}x{1}_{2}_{3}lr_run{4}_{5}.h5".format(model.size[0], model.size[1], aug_str, model.lr, run, arch)
        	model_path = train(model, model_str)
        	model_paths[arch].append(model_path)

	print("Done.")
	return model_paths
        
model_paths = train_all()

def test(model_paths):
	predictions_full = np.zeros((nb_test_samples, nb_classes))

	for run in range(nb_runs):
		print("\nStarting Prediction Run {0} of {1}...\n".format(run+1, nb_runs))
		predictions_aug = np.zeros((nb_test_samples, nb_classes))

		for aug in range(nb_aug):
			print("\n--Predicting on Augmentation {0} of {1}...\n".format(aug+1, nb_aug))
			predictions_mod = np.zeros((nb_test_samples, nb_classes))

			for arch in archs:
				print("----Predicting on {} model...".format(arch))
				parent = models[arch]
				model = parent.build()
				model.load_weights(model_paths[arch][run])
				pred, filenames = parent.test(test_path, nb_test_samples, aug=nb_aug)
				predictions_mod += pred

			predictions_mod /= len(archs)
			predictions_aug += predictions_mod

		predictions_aug /= nb_aug
		predictions_full += predictions_aug

	predictions_full /= nb_runs
	return predictions_full, filenames

predictions, filenames = test(model_paths)

def write_submission(predictions, filenames):
    preds = np.clip(predictions, clip, 1-clip)
    sub_fn = submission_path + '{0}epoch_{1}aug_{2}clip_{3}runs'.format(nb_epoch, nb_aug, clip, nb_runs)

    for arch in archs:
        sub_fn += "_{}".format(arch)

    with open(sub_fn + '.csv', 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for i, image_name in enumerate(filenames):
            pred = ['%.6f' % p for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
        print("Done.")

write_submission(predictions, filenames)