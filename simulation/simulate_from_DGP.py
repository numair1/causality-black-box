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

np.random.seed(0)


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
    os.mkdir("./data_full_pipeline")
    os.mkdir("./data_full_pipeline/1")
    os.mkdir("./data_full_pipeline/0")


def sample_z_and_y():
    """
    Function that generates binary valued samples that determine the shapes that
    go into the images
    :return: Dictionary of five ints representing binary indicators of the presence of shape in the image
    """
    y = np.random.binomial(n=1, p=0.5, size=1)
    circle = np.random.binomial(n=1, p= 0.25, size=1)
    v_bar = np.random.binomial(n=1, p=special.expit(0.3763 + 1.454 * y[0]), size=1)
    h_bar = np.random.binomial(n=1, p=special.expit(-0.687 - 1.03 * v_bar[0] + 1.069*y[0]), size=1)
    triangle = np.random.binomial(n=1, p=special.expit(-2 + 1.3*circle[0] + 2.2*h_bar[0]), size=1)
    return {"Y": int(y[0]), "C": int(circle[0]), "T": int(triangle[0]), "H": int(h_bar[0]), "V": int(v_bar[0])}


def create_img():
    """
    Function that creates a blank image that will later have shapes inserted

    :return: np.ndarray representing blank image
    """
    im_h_bar = np.random.binomial(n=1, p=0.01, size=(224, 224)) * 255
    im_v_bar = im_h_bar.copy()
    im_full = im_h_bar.copy()
    return im_h_bar, im_v_bar, im_full


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
        rr, cc = polygon((190, 190, 205), (190, 210, 200))
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
        im_h_bar, im_v_bar, im_full = create_img()
        im_features_and_label = sample_z_and_y()
        image_features_list.append(im_features_and_label)

        # Put in relevant shapes into images
        if im_features_and_label["H"] == 1:
            h_bar_pos = np.random.randint(20, 101, size=1)[0]
            insert_object_into_image(im_h_bar, "h_bar", h_bar_pos)
            insert_object_into_image(im_full, "h_bar", h_bar_pos)

        if im_features_and_label["V"] == 1:
            v_bar_pos = np.random.randint(20, 101, size=1)[0]
            insert_object_into_image(im_v_bar, "v_bar", v_bar_pos)
            insert_object_into_image(im_full, "v_bar", v_bar_pos)

        if im_features_and_label["C"] == 1:
            insert_object_into_image(im_full, "circle")

        if im_features_and_label["T"] == 1:
            insert_object_into_image(im_full, "triangle")

        im_h_bar = fix_image_format(im_h_bar)
        im_v_bar = fix_image_format(im_v_bar)
        im_full = fix_image_format(im_full)
        simulated_image_and_label_list.append({"Y": im_features_and_label["Y"],
                                               "im_h_bar": im_h_bar, "im_v_bar": im_v_bar,
                                               "im_full": im_full})
    return image_features_list, simulated_image_and_label_list


def write_outfile(feature_list):
    """
    Function that takes in the list of features for each image and writes them to a text file
    :param feature_list: List of dictionaries containing the features and labels for every image. Each dictionary in the
    list has the following format:
     - "Y": Label assigned to the image
     - "C": Indicator whether the image contains a circle
     - "T": Indicator whether the image contains a triangle
     - "H": Indicator whether the image contains a horizontal bar
     - "V": Indicator whether the image contains a vertical bar
     :param root_path: string containing the path to the directory where the data is stored
    :return: None
    """
    with open("./hand_crafted_ft.txt", "w+") as outfile:
        outfile.write("img_name\th_bar\tv_bar\tcircle\ttriangle\ty\n")
        i = 0
        for im_features in feature_list:
            im_name = "img_" + str(i) + ".png"
            outfile.write(im_name + "\t" + str(im_features["H"]) + "\t" + str(im_features["V"]) + "\t"
                          + str(im_features["C"]) + "\t" + str(im_features["T"]) + "\t" + str(
                im_features["Y"]) + "\n")
            i += 1


def save_images(image_list):
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
    i = 0
    for im in image_list:
        im_name = "img_" + str(i) + ".png"
        io.imsave("./data_full_pipeline/" + str(im["Y"]) + "/" + im_name, im["im_full"])
        i += 1

def simulate_images(n_obs):
    """
    Function that takes in simulation parameters as input and returns image feature annotations as well as simulated
    images along with the features
    :param n_obs: int representing number of images to generate
    :return: None
    """
    image_features_list, simulated_image_and_label_list = create_images(n_obs)
    make_directories()
    write_outfile(image_features_list)
    save_images(simulated_image_and_label_list)


if __name__ == "__main__":
    simulate_images(10000)
