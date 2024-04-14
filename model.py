import tensorflow as tf
import keras


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - score

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=1e-2)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def iou_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def r_squared(y_true, y_pred):
    """
    计算决定系数R^2，用作Keras模型的评估指标。

    :param y_true: 真实值
    :param y_pred: 预测值
    :return: R^2的值
    """
    # 总平方和（Total Sum of Squares, SSE）
    ss_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    # 回归平方和（Residual Sum of Squares, SSR）
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    # R^2 计算
    r2 = 1 - ss_res/ss_total
    return r2


def conv_layer(inputs, filters, kernel_size=3, strides=1, need_activate=True):
    out = keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(inputs)
    out = keras.layers.BatchNormalization()(out)
    if need_activate:
        out = keras.layers.ELU()(out)
    return out


def block_1(inputs, filters):
    out = conv_layer(inputs, filters // 4, 1)
    out = conv_layer(out, filters // 4, 3)
    out = conv_layer(out, filters // 4, 3, need_activate=False)
    res = conv_layer(inputs, filters // 4, 1)
    out = keras.layers.ELU()(out + res)
    return out


def block_2(inputs, filters):
    out = conv_layer(inputs, filters // 4, 1)
    out = conv_layer(out, filters // 4, 3)
    out = conv_layer(out, filters // 4, 3, need_activate=False)
    res = conv_layer(inputs, filters // 4, 1)
    out = keras.layers.ELU()(out + res)
    return out


def block_3(inputs, filters):
    out = conv_layer(inputs, filters // 4, 1)
    out = conv_layer(out, filters // 4, 3)
    out = conv_layer(out, filters, 3, need_activate=False)
    res = conv_layer(inputs, filters, 1)
    out = keras.layers.ELU()(out + res)
    return out


def dr_unet(pretrained_weights=None, input_size=(128, 128, 1), dims=32):
    inputs = keras.Input(input_size)
    out = conv_layer(inputs, 16, 1)

    out = block_1(out, dims)
    out_256 = block_3(out, dims)
    out = keras.layers.MaxPool2D(2, 2, padding='same')(out_256)

    out = block_1(out, dims * 2)
    out_128 = block_3(out, dims * 2)
    out = keras.layers.MaxPool2D(2, 2, padding='same')(out_128)

    out = block_1(out, dims * 4)
    out_64 = block_3(out, dims * 4)
    out = keras.layers.MaxPool2D(2, 2, padding='same')(out_64)

    out = block_1(out, dims * 8)
    out_32 = block_3(out, dims * 8)
    out = keras.layers.MaxPool2D(2, 2, padding='same')(out_32)

    out = block_1(out, dims * 16)
    out_16 = block_3(out, dims * 16)
    out = keras.layers.MaxPool2D(2, 2, padding='same')(out_16)

    out = block_1(out, dims * 32)
    out = block_3(out, dims * 32)

    up_16 = keras.layers.Conv2DTranspose(filters=dims * 16, kernel_size=2, strides=2, padding='same')(out)
    up = keras.layers.Concatenate()([up_16, out_16])
    up = keras.layers.BatchNormalization()(up)
    up = keras.layers.ELU()(up)

    up = block_2(up, dims * 16)
    up = block_3(up, dims * 16)
    up_32 = keras.layers.Conv2DTranspose(filters=dims * 8, kernel_size=2, strides=2, padding='same')(up)
    up = keras.layers.Concatenate()([up_32, out_32])
    up = keras.layers.BatchNormalization()(up)
    up = keras.layers.ELU()(up)

    up = block_2(up, dims * 8)
    up = block_3(up, dims * 8)
    up_64 = keras.layers.Conv2DTranspose(filters=dims * 4, kernel_size=2, strides=2, padding='same')(up)
    up = keras.layers.Concatenate()([up_64, out_64])
    up = keras.layers.BatchNormalization()(up)
    up = keras.layers.ELU()(up)

    up = block_2(up, dims * 4)
    up = block_3(up, dims * 4)
    up_128 = keras.layers.Conv2DTranspose(filters=dims * 2, kernel_size=2, strides=2, padding='same')(up)
    up = keras.layers.Concatenate()([up_128, out_128])
    up = keras.layers.BatchNormalization()(up)
    up = keras.layers.ELU()(up)

    up = block_2(up, dims * 2)
    up = block_3(up, dims * 2)
    up_256 = keras.layers.Conv2DTranspose(filters=dims * 1, kernel_size=2, strides=2, padding='same')(up)
    up = keras.layers.Concatenate()([up_256, out_256])
    up = keras.layers.BatchNormalization()(up)
    up = keras.layers.ELU()(up)

    up = block_2(up, dims)
    up = block_3(up, dims)
    up = keras.layers.Conv2D(filters=1, kernel_size=1, strides=(1, 1), padding='same')(up)
    up = keras.activations.sigmoid(up)

    model = keras.Model(inputs, up)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss=bce_dice_loss,
                  metrics=['accuracy', bce_dice_loss, dice_loss, iou_coefficient, r_squared])

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
        print("\nmodel load successfully!\n")
    return model


if __name__ == '__main__':
    import numpy as np

    def generate_fake_data(num_samples, input_size):
        # 生成随机数据作为样本
        x = np.random.random((num_samples,) + input_size)
        # 生成随机二进制标签作为样本标签
        y = np.random.randint(0, 2, (num_samples,) + input_size)
        return x, y


    num_samples = 100000
    input_size = (128, 128, 1)
    x_train, y_train = generate_fake_data(num_samples, input_size)

    model = dr_unet(pretrained_weights=None,
                    input_size=input_size)
    # model.fit(x_train, y_train, epochs=5, batch_size=1)
    batch_size = 16
    testPredictions = model.predict(x_train, verbose=1,
                                    batch_size=batch_size)


