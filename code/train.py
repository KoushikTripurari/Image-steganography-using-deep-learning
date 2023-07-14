from src.data_preparation import load_data, split_data, create_dataset
from src.model import create_models, train_model
from src.visualize import display_results
import argparse
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description='Image steganography Using Deep Learning, train on the cover images and secret images.')
    parser.add_argument('--cv_path', type=str, help='The path to the cover images folder.')
    parser.add_argument('--sc_path', type=str, help='The path to the secret images folder.')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')

    args = parser.parse_args()

    # Load data
    cover_images, secret_images = load_data(args.cv_path, args.sc_path)

    # Split data into training and validation sets
    X_cover_train, X_cover_val, X_secret_train, X_secret_val = split_data(cover_images, secret_images)

    # Define hyperparameters
    buffer_size = 1000

    # Create datasets
    train_dataset = create_dataset(X_cover_train, X_secret_train, args.batch_size, buffer_size)
    val_dataset = create_dataset(X_cover_val, X_secret_val, args.batch_size, buffer_size)

    # Defining inputs for the combined model
    cover_image_input = tf.keras.layers.Input(shape= (64, 64, 3), name='secret_image_input')
    secret_image_input = tf.keras.layers.Input(shape= (64, 64, 3), name='cover_image_input')
    hidden_image_input = tf.keras.layers.Input(shape=(64, 64, 3), name="hidden_image_input")

    # Create models
    encoder_model, decoder_model, combined_model = create_models()

    # Train model
    train_model(combined_model, train_dataset, val_dataset, args.epochs)

    encoder_model.save('models/encoder_model.h5')
    decoder_model.save('models/decoder_model.h5')
    combined_model.save('models/combined_model.h5')

    # Display results
    display_results(val_dataset, encoder_model, decoder_model)

if __name__ == "__main__":
    main()
