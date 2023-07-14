import matplotlib.pyplot as plt

def display_results(val_dataset, encoder_model, decoder_model, num_examples=5):
    cover_batch, secret_batch = next(iter(val_dataset))[0]
    hidden_batch = encoder_model.predict([cover_batch, secret_batch])
    revealed_batch = decoder_model.predict(hidden_batch)
    
    # Display the images as per your requirements
    fig, axes = plt.subplots(num_examples, 4, figsize=(20, num_examples * 5))

    for i in range(num_examples):
        axes[i, 0].imshow(cover_batch[i])
        axes[i, 0].set_title("Cover Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(secret_batch[i])
        axes[i, 1].set_title("Secret Image")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(hidden_batch[i])
        axes[i, 2].set_title("Hidden Image")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(revealed_batch[i])
        axes[i, 3].set_title("Revealed Image")
        axes[i, 3].axis("off")
    plt.savefig('results/result.jpg')
    plt.show()