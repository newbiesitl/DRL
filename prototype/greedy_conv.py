from keras.layers import Dense
from keras.models import Sequential, Model
from keras.optimizers import Adadelta
from keras.losses import binary_crossentropy
from utils.utils import *
from embedding.model_config import *
from keras.callbacks import TensorBoard
cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..')
data_folder = os.path.join(project_root, 'data', 'images', 'men')

train_model = True

t =ImageTransformer()
t.configure(input_shape)
train = None
test = None
strides = (1,1)
for dat in t.transform_all(data_folder, grey_scale=False, batch_size=10 if not train_model else 10000):
    if train is None:
        train = dat # get the first batch from generator
        continue
    if test is None:
        test = dat
        continue
    break

test = train[int(len(train)*0.8):]
train = train[:int(len(train)*0.8)]
print(train.shape)
print(test.shape)

ae = Sequential()
x = Conv2D(16, (3, 3), activation=activation_1, padding='same', input_shape=input_shape)
ae.add(x)
x = MaxPooling2D((2, 2))
ae.add(x)
x = Conv2D(12, (3, 3), activation=activation_1, padding='same')
ae.add(x)
x = MaxPooling2D((2, 2))
ae.add(x)

x = Conv2D(12, (3, 3), activation=activation_1, padding='same')
ae.add(x)
x = MaxPooling2D((2, 2))
ae.add(x)

x = Conv2D(12, (3, 3), activation=activation_1, padding='same')
ae.add(x)
x = UpSampling2D((2, 2))
ae.add(x)


# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = Conv2D(12, (3, 3), activation=activation_1, padding='same')
ae.add(x)
x = UpSampling2D((2, 2))
ae.add(x)

x = Conv2D(16, (3, 3), activation=activation_1, padding='same')
ae.add(x)
x = UpSampling2D((2, 2))
ae.add(x)
x = Conv2D(3, (3, 3), activation=activation_2, padding='same')


ae.add(x)


ae.compile(optimizer='adadelta', loss='binary_crossentropy')
ae.fit(train, train,
                epochs=12,
                batch_size=128,
                shuffle=True,
                validation_data=(test, test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
visualize_result_ae(ae, test, input_shape, color_img=True)
visualize_result_ae(ae, train, input_shape, color_img=True)