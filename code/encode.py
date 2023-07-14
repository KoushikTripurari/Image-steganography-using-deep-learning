import argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Testing encoder model using cover and secret images.')
    parser.add_argument('--cv_path', type=str, help='The path to the test cover image.')
    parser.add_argument('--sc_path', type=str, help='The path to the test secret image.')
    parser.add_argument('--encoder_path', type=str, help='The path to encoder model path.')

    args = parser.parse_args()

    cover_img_path = args.cv_path
    secret_img_path = args.sc_path
    encoder_model_path = args.encoder_path

    # Load the model
    encoder_model = load_model(encoder_model_path)

    # Preprocessing Images
    cover_image = Image.open(cover_img_path).convert('RGB').resize((64,64))
    secret_image = Image.open(secret_img_path).convert('RGB').resize((64,64))

    cover_image = np.array(cover_image)/255.0
    secret_image = np.array(secret_image)/255.0

    # Adding an extra dimension
    cover_image = np.expand_dims(cover_image, axis=0) 
    secret_image = np.expand_dims(secret_image, axis=0)
    
    # Encode
    hidden_image = encoder_model.predict([cover_image, secret_image])
    
    #Displaying image
    plt.imshow(np.squeeze(hidden_image, axis = 0))
    plt.savefig('results/encoded_img.jpg')

if __name__ == "__main__":
    main()