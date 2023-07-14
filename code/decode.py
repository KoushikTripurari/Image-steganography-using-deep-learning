import argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Testing the decoder using encoded images.')
    parser.add_argument('--en_img_path', type=str, help='The path to the encoded image.')
    parser.add_argument('--decoder_path', type=str, help='The path to decoder model path.')

    args = parser.parse_args()

    encoded_img_path = args.en_img_path
    decoder_model_path = args.decoder_path

    # Load the model
    decoder_model = load_model(decoder_model_path)

    # Preprocessing Images
    encoded_img = Image.open(encoded_img_path).convert('RGB').resize((64,64))
    encoded_img = cover_image = np.array(encoded_img)/255.0

    # Adding an extra dimension
    encoded_img = np.expand_dims(encoded_img, axis=0)
    
    # Encode
    reveal_image = decoder_model.predict(encoded_img)
    
    #Displaying image
    plt.imshow(np.squeeze(reveal_image, axis = 0))
    plt.savefig('results/reveal_img.jpg')

if __name__ == "__main__":
    main()