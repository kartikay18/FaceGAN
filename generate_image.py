from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import resize_face
import numpy as np
from PIL import Image
import argparse
import math
import theano



def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model


def load_image(filepath):
    img = Image.open(filepath).convert('L')
    img = resizeimage.resize_cover(img, [28, 28], validate=False)
    return img


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image



def generate(count, generator, BATCH_SIZE=1):
    print "Entered generator"
    noise = np.zeros((BATCH_SIZE, 100))
    for i in range(BATCH_SIZE):
        noise[i, :] = np.random.uniform(-1, 1, 100)
    generated_images = generator.predict(noise, verbose=1)
    image = combine_images(generated_images)
    image = image*127.5+127.5
    final_filename_png = "generated_data/generated_image_" + str(count) + ".png"
    final_filename_npy = "generated_data/generated_noise_" + str(count) + ".npy"
    Image.fromarray(image.astype(np.uint8)).save(
        final_filename_png)
    np.save(final_filename_npy, noise[0, :])
    print "Saved generated_image.png and generated_noise.py"
    return image



def generate_from_array(array_filename, BATCH_SIZE=1):
    print "Entered generator_from_array"
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator_faces')
    noise = np.zeros((BATCH_SIZE, 100))
    noise[0, :] = np.load(array_filename)
    #for i in range(BATCH_SIZE):
    #    noise[i, :] = np.random.uniform(-1, 1, 100)
    generated_images = generator.predict(noise, verbose=1)
    image = combine_images(generated_images)
    image = image*127.5+127.5
    print "Saved generated_image.png"
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image_from_array.png")
    np.save('generated_noise_from_array.npy', noise[0, :])
    return image


def load_from_array_mode():
    array_filename = "generated_noise.npy"
    generate_from_array(array_filename)


def batch_generate_mode(number_of_images):

    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator_faces')

    for i in range(0, number_of_images):
        generate(i, generator)        




if __name__ == "__main__":
    #filepath = "input.jpg"
    #input_image = load_image(filepath)
    #generated_image = generate()
    
    batch_generate_mode(30)

    #generate_from_array('generated_data/generated_noise_1.npy')
    #generate_from_array('generated_noise_1.npy')
    #load_from_array_mode()


    '''
    mask = np.ones(input_image.shape)
    mask_size = 7
    mask_coord = 13
    #w = input_image.shape[1]
    mask[mask_coord:(mask_coord+mask_size), mask_coord:(mask_coord+mask_size)] = 0

    masked_image = np.multiply(input_image, mask)
    Image.fromarray(masked_image.astype(np.uint8)).save(
        "masked_image.png")
    '''




