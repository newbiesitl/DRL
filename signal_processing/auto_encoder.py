from keras import layers, models



class AE(object):
    def __init__(self, i_dim=None, h_dim=None):
        if i_dim is None or h_dim is None:
            # this is to load model
            return
        self.m = models.Sequential()
        layer = layers.Dense(i_dim, input_shape=(i_dim,))
        self.m.add(layer)
        layer = layers.Dense(h_dim, activation='sigmoid')
        self.m.add(layer)
        layer = layers.Dense(i_dim, activation='linear')
        self.m.add(layer)
        self.m.compile(optimizer='adam', loss='mean_absolute_error')

    def fit(self, X, batch_size=20, epoches=5):
        self.m.fit(X, X, batch_size=batch_size, epochs=epoches)

    def save(self, path):
        self.m.save(path+'.h5', overwrite=True, include_optimizer=True)

    def load(self, path):
        self.m = models.load_model(path+'.h5')

    def predict(self, X):
        return self.m.predict(X)