from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from os.path import join as pjoin
import config

def dnn_model(nn):
    # Build model
    # preprocessing = layers.experimental.preprocessing
    deep_diffusion = keras.models.Sequential()
    deep_diffusion.add(layers.Dense(nn, input_dim=2, activation='elu'))
    for n in range(config.nlayer):
        deep_diffusion.add(layers.Dense(nn, activation='elu'))
    # deep_diffusion.add(layers.experimental.preprocessing.RandomRotation(factor = 0.2))
    # deep_diffusion.add(layers.experimental.preprocessing.RandomWidth(factor = 0.2))
    # deep_diffusion.add(layers.experimental.preprocessing.RandomHeight(factor = 0.2))
    # deep_diffusion.add(layers.experimental.preprocessing.RandomZoom(factor = 0.2))
    # deep_diffusion.add(layers.experimental.preprocessing.RandomFlip(mode='horizontal'))

    deep_diffusion.add(layers.Dense(1, activation='linear'))

    # Compile model
    deep_diffusion.compile(loss='mse', optimizer='adam')
    return deep_diffusion

def train_dnn(deep_diffusion,inputs_array,outputs_array,savedir):
    ########
    test_ratio = 0.25
    dev_ratio = 0.2

    # Split into train-dev-test sets
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(inputs_array, outputs_array, test_size=test_ratio, shuffle=False)
    Xs_train, Xs_dev, ys_train, ys_dev = train_test_split(Xs_train, ys_train, test_size=dev_ratio, shuffle=True)

    # Fit!
    history = deep_diffusion.fit(Xs_train, ys_train, epochs=config.epochs, batch_size=config.batch_size,
                validation_data=(Xs_dev, ys_dev),
                callbacks=keras.callbacks.EarlyStopping(patience=config.patience))

    deep_diffusion.summary()
    deep_diffusion.save(pjoin(savedir),overwrite=True,include_optimizer=True,save_format=None)
    return deep_diffusion,history
