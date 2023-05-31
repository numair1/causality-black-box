import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from skimage.draw import disk
from skimage.draw import rectangle
from scipy import special
from skimage.draw import polygon
import os


np.random.seed(1)


def fix_image_format(im):
    """
    Function that takes in image represented by np.ndarray and converts its
    values to np.uint8

    :param im: np.ndarray representing an image
    return: np.ndarray in proper format
    """
    im = color.gray2rgb(im)
    im = im.astype(np.uint8)
    return im


def make_directories():
    """
    Function that makes empty placeholder directories  for the images to
    be saved in

    :return: None
    """
    os.mkdir("./val_data")


def sample_image_objects():
    """
    Function that generates binary valued samples that determine the shapes that
    go into the images
    :return: Dictionary of five ints representing binary indicators of the presence of shape in the image
    """
    circle = np.random.binomial(n=1, p= 0.5, size=1)
    v_bar = np.random.binomial(n=1, p=0.5, size=1)
    h_bar = np.random.binomial(n=1, p=0.5, size=1)
    triangle = np.random.binomial(n=1, p=0.5, size=1)
    return {"C": int(circle[0]), "T": int(triangle[0]), "H": int(h_bar[0]), "V": int(v_bar[0])}


def create_img():
    """
    Function that creates a blank image that will later have shapes inserted

    :return: np.ndarray representing blank image
    """
    im = np.random.binomial(n=1, p=0.01, size=(224, 224)) * 255
    return im


def insert_object_into_image(im, object_to_insert, pos=None):
    """
    Function that takes an image and corresponding object to insert and inserts
    it at either a random position or fixed psoition depending on object

    :param im: np.ndarray representing an image
    :param object_to_insert: str representing which object h-bar, v-bar, circle, triangle
    to insert
    :param pos: int representing where in the image to insert a h-bar or a v-bar. Only for these two objects as the
    rest go in fixed places
    :return: np.ndarray containing inserted object
    """
    if object_to_insert == "h_bar":
        im[pos: pos + 15, :] = 255

    elif object_to_insert == "v_bar":
        im[:, pos: pos + 15] = 255

    elif object_to_insert == "circle":
        rr, cc = disk((150, 150), 15)
        im[rr, cc] = 255

    elif object_to_insert == "triangle":
        rr, cc = polygon((190, 190, 210), (190, 220, 205))
        im[rr, cc] = 255

    return im


def create_images(n_obs):
    """
    Function that generates the images according to simulation parameters specified

    :param n_obs: Int representing number of integers
    :return: Two list of Dicts, one representing the handcrafted feature annotations for each image
    and the other storing the various versions of images and the corresponding label
    """
    image_features_list = []
    simulated_image_and_label_list = []
    for i in range(n_obs):
        im = create_img()
        im_objects = sample_image_objects()
        image_features_list.append(im_objects)

        # Put in relevant shapes into images
        if im_objects["H"] == 1:
            h_bar_pos = np.random.randint(20, 101, size=1)[0]
            insert_object_into_image(im, "h_bar", h_bar_pos)

        if im_objects["V"] == 1:
            v_bar_pos = np.random.randint(20, 101, size=1)[0]
            insert_object_into_image(im, "v_bar", v_bar_pos)

        if im_objects["C"] == 1:
            insert_object_into_image(im, "circle")

        if im_objects["T"] == 1:
            insert_object_into_image(im, "triangle")

        im = fix_image_format(im)
        save_image(im, im_objects, i)


def save_image(im, im_objects, im_index):
    """
    Function that takes in list of images and saves each one in the relevant directory, with an 80-20 train validation
    split for im_h_bar and im_v_bar and no split for im_full as this is only used for validation

    :param image_list: List of dicts with the following format for each element
    - "Y": Image label
    - "im_h_bar": Image with just the horizontal bar,
    - "im_v_bar": Image with just the vertical bar,
    - "im_full": Full image
    :return: None
    """
    feature_annotation_str = str(im_objects["C"]) + "_" + str(im_objects["H"]) + "_" + str(im_objects["T"])\
                             + "_" + str(im_objects["V"])
    im_name = feature_annotation_str + "_" + "img_" + str(im_index) + ".png"
    io.imsave("./val_data/" + im_name, im)


def simulate_images(n_obs):
    """
    Function that takes in simulation parameters as input and returns image feature annotations as well as simulated
    images along with the features
    :param n_obs: int representing number of images to generate
    :return: None
    """
    make_directories()
    create_images(n_obs)

if __name__ == "__main__":
    simulate_images(500)
