from keras.layers import Layer
from keras import backend as K
from keras import initializers, regularizers, constraints


class Attention(Layer):
    """
    Describe:
        Input : (samples,steps,fetures)
        Output:(samples,steps,fetures)
    """

    def __init__(self, categeory_count, is_last_layer=False, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None, **kwargs):
        self.categeroy_count = categeory_count
        self.is_last_layer = is_last_layer
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.b_constraint = regularizers.get(b_constraint)

        self.init = initializers.get("glorot_uniform")

        self.steps_dim = 0
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(
            input_shape) == 3, "Att's input shape is {}.You'd turn shape into (samples,steps,features)".format(
            input_shape)

        self.steps_dim = input_shape[1]
        self.features_dim = input_shape[2]

        self.WQ = self.add_weight(name="WQ", shape=(self.features_dim, self.features_dim),
                                  initializer=self.init, trainable=True)
        self.label_emb = self.add_weight(name="Label", shape=(self.categeroy_count, self.features_dim),
                                         initializer=self.init, trainable=True)
        self.WK = self.add_weight(name="WK", shape=(self.features_dim, self.features_dim),
                                  initializer=self.init, trainable=True)
        if self.is_last_layer is False:
            self.WV = self.add_weight(name="WV", shape=(self.features_dim, self.features_dim),
                                      initializer=self.init, trainable=True)
        self.built = True

    # TODO 这里没有去实践Mask和多头的功能，所以比较简单

    def call(self, inputs, **kwargs):
        Q_ = K.dot(inputs, self.WQ)
        K_ = K.dot(self.label_emb, self.WK)

        # step1: 计算S = Q_* K_
        K_ = K.permute_dimensions(K_, [1, 0])
        S_ = K.dot(Q_, K_)
        # step2: 计算类softMax归一化
        A_ = K.softmax(S_, axis=-1)
        if self.is_last_layer is False:
            # step3: 计算 带权V_值
            V_ = K.dot(self.label_emb, self.WV)
            return K.dot(A_, V_)
        else:
            return A_

    def compute_output_shape(self, input_shape):
        if self.is_last_layer is False:
            return input_shape[0], self.steps_dim, self.features_dim
        else:
            return input_shape[0], self.steps_dim, self.categeroy_count
