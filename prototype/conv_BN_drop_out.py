from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adadelta, Adam
from keras.losses import binary_crossentropy
from utils.utils import *
from embedding.model_config import *
from keras.callbacks import TensorBoard
import shutil

'''
Use TensorBoard
1. run in command line `tensorboard --logdir path_to_log_dir`
2. http://localhost:6006/
'''

cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..')
data_folder = os.path.join(project_root, 'data', 'images', 'all')
tensor_board_path = '/tmp/exp1'
try:
    shutil.rmtree(tensor_board_path)
except:
    pass
train_model = False
model_folder = os.path.join(project_root, 'models', 'vision')
model_name = 'bn_conv_all'
t =ImageTransformer()
t.configure(input_shape)
train = None
test = None
num_epoch = 3
strides = (1,1)
for dat in t.transform_all(data_folder, grey_scale=False, batch_size=100 if not train_model else 50000):
    if train is None:
        train = dat # get the first batch from generator
        continue
    if test is None:
        test = dat
        continue
    break

test = train[int(len(train)*0.9):]
train = train[:int(len(train)*0.9)]
print(train.shape)
print(test.shape)

bn = True
use_dropout = False

if train_model:
    ae = Sequential()
    x = Conv2D(16, (5, 5), activation=activation_1, padding='same', input_shape=input_shape)
    ae.add(x)
    if bn:
        ae.add(BatchNormalization())
    if use_dropout:
        ae.add(Dropout(0.5))
    x = MaxPooling2D((2, 2))
    ae.add(x)
    x = Conv2D(32, (5, 5), activation=activation_1, padding='same')
    ae.add(x)
    if bn:
        ae.add(BatchNormalization())
    if use_dropout:
        ae.add(Dropout(0.5))
    x = MaxPooling2D((2, 2))
    ae.add(x)

    x = Conv2D(64, (3, 3), activation=activation_1, padding='same')
    ae.add(x)
    if bn:
        ae.add(BatchNormalization())
    if use_dropout:
        ae.add(Dropout(0.5))
    x = MaxPooling2D((2, 2))
    ae.add(x)

    x = Conv2D(64, (3, 3), activation=activation_1, padding='same')
    ae.add(x)
    if bn:
        ae.add(BatchNormalization())
    if use_dropout:
        ae.add(Dropout(0.5))
    x = UpSampling2D((2, 2))
    ae.add(x)


    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Conv2D(32, (5, 5), activation=activation_1, padding='same')
    ae.add(x)
    if bn:
        ae.add(BatchNormalization())
    if use_dropout:
        ae.add(Dropout(0.5))
    x = UpSampling2D((2, 2))
    ae.add(x)

    x = Conv2D(16, (5, 5), activation=activation_1, padding='same')
    ae.add(x)
    if bn:
        ae.add(BatchNormalization())
    if use_dropout:
        ae.add(Dropout(0.5))
    x = UpSampling2D((2, 2))
    ae.add(x)
    x = Conv2D(3, (3, 3), activation=activation_2, padding='same')
    ae.add(x)


    ae.compile(optimizer=Adam(), loss='binary_crossentropy')
    ae.fit(train, train,
           epochs=num_epoch,
           batch_size=128,
           shuffle=True,
           validation_data=(test, test),
           callbacks=[TensorBoard(log_dir=tensor_board_path)])
    # serialize model to JSON
    model_json = ae.to_json()
    with open(os.path.join(model_folder, model_name+".json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    ae.save_weights(os.path.join(model_folder, model_name+".h5"))
    print("Saved model to disk")
else:
    json_file = open(os.path.join(model_folder, model_name+'.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    ae = model_from_json(loaded_model_json)
    # load weights into new model
    ae.load_weights(os.path.join(model_folder, model_name+".h5"))
    print("Loaded model from disk")
visualize_result_ae(ae, test, input_shape, color_img=True)
visualize_result_ae(ae, train, input_shape, color_img=True)