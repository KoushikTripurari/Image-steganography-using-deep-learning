import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def prep_network(secret_image):
    conv1 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3,3), strides=1, padding='same')
    conv2 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same')
    conv3 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5,5), strides=1, padding='same')

    conv4 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3,3), strides=1, padding='same')
    conv5 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same')
    conv6 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5,5), strides=1, padding='same')

    output_1 = tf.nn.relu(conv1(secret_image))
    output_2 = tf.nn.relu(conv2(secret_image))
    output_3 = tf.nn.relu(conv3(secret_image))

    concatenated_image = tf.concat([output_1, output_2, output_3], axis=3)
    
    output_4 = tf.nn.relu(conv4(concatenated_image))
    output_5 = tf.nn.relu(conv5(concatenated_image))
    output_6 = tf.nn.relu(conv6(concatenated_image))

    final_concat_image = tf.concat([output_4, output_5, output_6], axis=3)
    return final_concat_image

def hiding_network(secret_image_1, cover_image):
    conv1 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), padding='same')
    conv2 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1), padding='same')
    conv3 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5, 5), strides=(1, 1), padding='same')

    conv4 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), padding='same')
    conv5 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1), padding='same')
    conv6 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5, 5), strides=(1, 1), padding='same')

    conv7 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), padding='same')
    conv8 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1), padding='same')
    conv9 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5, 5), strides=(1, 1), padding='same')

    conv10 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), padding='same')
    conv11 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1), padding='same')
    conv12 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5, 5), strides=(1, 1), padding='same')

    conv13 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), strides=(1, 1), padding='same')
    conv14 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1), padding='same')
    conv15 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5, 5), strides=(1, 1), padding='same')

    final_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same')

    concatenated_secrets = tf.concat([cover_image, secret_image_1], axis=3)

    output_1 = tf.nn.relu(conv1(concatenated_secrets))
    output_2 = tf.nn.relu(conv2(concatenated_secrets))
    output_3 = tf.nn.relu(conv3(concatenated_secrets))
    concat_1 = tf.concat([output_1, output_2, output_3], axis=3)

    output_4 = tf.nn.relu(conv4(concat_1))
    output_5 = tf.nn.relu(conv5(concat_1))
    output_6 = tf.nn.relu(conv6(concat_1))
    concat_2 = tf.concat([output_4, output_5, output_6], axis=3)

    output_7 = tf.nn.relu(conv7(concat_2))
    output_8 = tf.nn.relu(conv8(concat_2))
    output_9 = tf.nn.relu(conv9(concat_2))
    concat_3 = tf.concat([output_7, output_8, output_9], axis=3)

    output_10 = tf.nn.relu(conv10(concat_3))
    output_11 = tf.nn.relu(conv11(concat_3))
    output_12 = tf.nn.relu(conv12(concat_3))
    concat_4 = tf.concat([output_10, output_11, output_12], axis=3)

    output_13 = tf.nn.relu(conv13(concat_4))
    output_14 = tf.nn.relu(conv14(concat_4))
    output_15 = tf.nn.relu(conv15(concat_4))
    concat_5 = tf.concat([output_13, output_14, output_15], axis=3)

    output_converted_image = tf.nn.relu(final_layer(concat_5))

    return output_converted_image

def encoder(secret_image, cover_image):
    # Process the secret image using the preparation network
    prepped_secret_image = prep_network(secret_image)

    # Combine the prepped secret image and cover image using the hiding network
    encoded_image = hiding_network(prepped_secret_image, cover_image)

    return encoded_image


def reveal_network(hidden_image):
    conv1 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3,3), strides=1, padding='same')
    conv2 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same')
    conv3 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5,5), strides=1, padding='same')

    conv4 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3,3), strides=1, padding='same')
    conv5 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same')
    conv6 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5,5), strides=1, padding='same')

    conv7 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3,3), strides=1, padding='same')
    conv8 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same')
    conv9 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5,5), strides=1, padding='same')

    conv10 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3,3), strides=1, padding='same')
    conv11 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same')
    conv12 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5,5), strides=1, padding='same')

    conv13 = tf.keras.layers.Conv2D(filters=50, kernel_size=(3,3), strides=1, padding='same')
    conv14 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same')
    conv15 = tf.keras.layers.Conv2D(filters=5, kernel_size=(5,5), strides=1, padding='same')

    final_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same')

    output_1 = tf.nn.relu(conv1(hidden_image))
    output_2 = tf.nn.relu(conv2(hidden_image))
    output_3 = tf.nn.relu(conv3(hidden_image))
    concat_1 = tf.concat([output_1, output_2, output_3], axis=3)

    output_4 = tf.nn.relu(conv4(concat_1))
    output_5 = tf.nn.relu(conv5(concat_1))
    output_6 = tf.nn.relu(conv6(concat_1))
    concat_2 = tf.concat([output_4, output_5, output_6], axis=3)

    output_7 = tf.nn.relu(conv7(concat_2))
    output_8 = tf.nn.relu(conv8(concat_2))
    output_9 = tf.nn.relu(conv9(concat_2))
    concat_3 = tf.concat([output_7, output_8, output_9], axis=3)

    output_10 = tf.nn.relu(conv10(concat_3))
    output_11 = tf.nn.relu(conv11(concat_3))
    output_12 = tf.nn.relu(conv12(concat_3))
    concat_4 = tf.concat([output_10, output_11, output_12], axis=3)

    output_13 = tf.nn.relu(conv13(concat_4))
    output_14 = tf.nn.relu(conv14(concat_4))
    output_15 = tf.nn.relu(conv15(concat_4))
    concat_5 = tf.concat([output_13, output_14, output_15], axis=3)

    output_revealed_image = tf.nn.relu(final_layer(concat_5))

    return output_revealed_image

def decoder(hidden_image):
  return reveal_network(hidden_image)

def create_models():
    # Define the input tensors for secret and cover images
    secret_image_input = tf.keras.layers.Input(shape=(64, 64, 3), name='secret_image_input')
    cover_image_input = tf.keras.layers.Input(shape=(64, 64, 3), name='cover_image_input')

    # Use the encoder function to create the encoded image
    encoded_image_output = encoder(secret_image_input, cover_image_input)

    # Create the encoder model
    encoder_model = tf.keras.Model(inputs=[secret_image_input, cover_image_input], outputs=encoded_image_output, name='encoder_model')

    # Define the input layer for the hidden image
    hidden_image_input = tf.keras.layers.Input(shape=(64, 64, 3), name="hidden_image_input")

    # Use the reveal network to create the revealed image
    revealed_image_output = reveal_network(hidden_image_input)

    # Create the decoder model
    decoder_model = tf.keras.Model(inputs=hidden_image_input, outputs=revealed_image_output, name="decoder_model")

    # Define input layers for secret and cover images for the combined model
    combined_secret_image_input = tf.keras.layers.Input(shape=(64, 64, 3), name="combined_secret_image_input")
    combined_cover_image_input = tf.keras.layers.Input(shape=(64, 64, 3), name="combined_cover_image_input")

    # Use the encoder model to create the encoded (hidden) image
    combined_encoded_image_output = encoder_model([combined_secret_image_input, combined_cover_image_input])

    # Use the decoder model to create the revealed (decoded) image from the encoded image
    combined_revealed_image_output = decoder_model(combined_encoded_image_output)

    # Create the combined model
    combined_model = tf.keras.Model(inputs=[combined_secret_image_input, combined_cover_image_input], outputs=[combined_encoded_image_output, combined_revealed_image_output], name="combined_model")

    return encoder_model, decoder_model, combined_model



def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def train_model(combined_model, train_dataset, val_dataset, epochs):
    combined_model.compile(optimizer='adam', loss=[custom_loss, custom_loss], loss_weights=[0.5, 0.5])
    combined_model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
