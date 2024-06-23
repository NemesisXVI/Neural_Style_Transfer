import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
import pprint

img_size = 400
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((img_size, img_size))
    img = np.array(img)
    return img

def run(content_image_input,style_image_input,epochs):
    tf.random.set_seed(272) # DO NOT CHANGE THIS VALUE
    pp = pprint.PrettyPrinter(indent=4)

    vgg = tf.keras.applications.VGG19(include_top=False,
                                      input_shape=(img_size, img_size, 3),
                                      weights='imagenet')

    vgg.trainable = False
    pp.pprint(vgg)
    content_image = content_image_input
    style_image = style_image_input

    def compute_content_cost(content_output, generated_output):
        """
        Computes the content cost

        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

        Returns:
        J_content -- scalar that you compute using equation 1 above.
        """
        a_C = content_output[-1]
        a_G = generated_output[-1]

        # Retrieving dimensions from a_G
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshaping 'a_C' and 'a_G'
        a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
        a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])

        # computing the Content Cost with tensorflow (≈1 line)
        J_content = tf.reduce_sum(tf.square(a_C_unrolled-a_G_unrolled)/(4*n_H*n_W*n_C))

        return J_content


    def gram_matrix(A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)

        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """
        GA = tf.matmul(A,tf.transpose(A))

        return GA


    def compute_layer_style_cost(a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

        Returns:
        J_style_layer -- tensor representing a scalar value, style cost defined.
        """

        # Retrieving dimensions from a_G
        _, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshaping the tensors from (1, n_H, n_W, n_C) to (n_C, n_H * n_W)
        a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
        a_G =  tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))

        # Computing gram_matrices for both images S and G
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)

        # Computing the loss
        J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GG,GS)))/(2*n_C*n_H*n_W)**2


        return J_style_layer

    for layer in vgg.layers:
        print(layer.name)

    STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)]

    """How to choose the coefficients for each layer:
     The deeper layers capture higher-level concepts,
     and the features in the deeper layers are less localized in the image relative to each other. 
     So if you want the generated image to softly follow the style image, try choosing larger weights for deeper layers and smaller weights for the first layers. 
     In contrast, if you want the generated image to strongly follow the style image, 
     try choosing smaller weights for deeper layers and larger weights for the first layers."""

    def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
        """
        Computes the overall style cost from several chosen layers

        Arguments:
        style_image_output --  tensorflow model
        generated_image_output --
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would extract style from
                            - a coefficient for each of them

        Returns:
        J_style -- tensor representing a scalar value, style cost defined.
        """

        # initializing the overall style cost to zero
        J_style = 0

        #Setting  a_S to be the hidden layer activation from the layer we have selected.
        # The last element of the array contains the content layer image, which must not be used.
        a_S = style_image_output[:-1]

        # Setting a_G to be the output of the chosen hidden layers.
        # The last element of the list contains the content layer image which must not be used.
        a_G = generated_image_output[:-1]

        for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
            # Computing style_cost for the current layer
            J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

            # Add weight * J_style_layer of this layer to overall style cost
            J_style += weight[1] * J_style_layer

        return J_style


    @tf.function()
    def total_cost(J_content, J_style, alpha = 10, beta = 40):
        """
        Computes the total cost function

        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost

        Returns:
        J -- total cost as defined by the formula above.
        """

        J = alpha*J_content + beta*J_style

        return J

    #Finally, putting everything together to implement Neural Style Transfer!

    """Here's the flow of the program:
    
    1. Load the content image
    2. Load the style image
    3. Randomly initialize the image to be generated
    4. Load the VGG19 model
    5. Compute the content cost
    6. Compute the style cost
    7. Compute the total cost
    8. Define the optimizer and learning rate"""

    # Setting up the Content Image and Style Image
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
    # Setting up the Generated Image and adding Noise to it.

    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

    #Loading the VGG19 model
    def get_layer_outputs(vgg, layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    content_layer = [('block5_conv4', 1)]

    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
    content_target = vgg_model_outputs(content_image)  # Content encoder
    style_targets = vgg_model_outputs(style_image)     # Style encoder

    # Assigning the content image to be the input of the VGG model.
    # Set a_C to be the hidden layer activation from the layer we have selected
    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)

    # Assigning the input of the model to be the "style" image
    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)


    def tensor_to_image(tensor):
        """
        Converts the given tensor into a PIL image

        Arguments:
        tensor -- Tensor

        Returns:
        Image: A PIL image
        """
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)


    # Using the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function()
    def train_step(generated_image):
        with tf.GradientTape() as tape:
            # Computing a_G as the vgg_model_outputs for the current generated image
            a_G = vgg_model_outputs(generated_image)

            # Computing the style cost
            J_style = compute_style_cost(a_S, a_G)

            # Computing the content cost
            J_content = compute_content_cost(a_C, a_G)
            # Computing the total cost
            J = total_cost(J_content, J_style)


        grad = tape.gradient(J, generated_image)

        optimizer.apply_gradients([(grad, generated_image)])

        return J


    generated_image = tf.Variable(generated_image)

    # Show the generated image at some epochs
    for i in range(epochs):
        train_step(generated_image)
        if i % 2 == 0:
            print(f"Epoch {i} ")
        # if i % 2 == 0:
        #     image = tensor_to_image(generated_image)
        #     imshow(image)
        #     plt.show()
    gen_img = tensor_to_image(generated_image)
    return gen_img




if __name__ == "__main__":
    # Setting example content and style image inputs
    content_image_input = load_image("images/Louvre_Museum.jpg")
    style_image_input = load_image("images/painting-impressionist-style.jpg")
    # running the model to obtain the generated image
    epochs=11
    generated_image = run(content_image_input,style_image_input,epochs)

    # Plotting the Content image,Style image and Generated Image side by side.
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    imshow(content_image_input)
    ax.title.set_text('Content image')
    ax = fig.add_subplot(1, 3, 2)
    imshow(style_image_input)
    ax.title.set_text('Style image')
    ax = fig.add_subplot(1, 3, 3)
    imshow(generated_image)
    ax.title.set_text('Generated image')
    plt.show()