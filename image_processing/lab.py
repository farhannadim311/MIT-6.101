#!/usr/bin/env python3

"""
6.101 Lab:
Image Processing
"""

import math
import os
from PIL import Image # type: ignore

# NO ADDITIONAL IMPORTS ALLOWED!

def oned_to_twod(image):
    twod_image = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [[0 for _ in range(image["width"])] for _ in range(image["height"])]
    }
    index = 0
    for col in range(image["height"]):
        for row in range(image["width"]):
            twod_image["pixels"][col][row] = image["pixels"][index]
            index += 1
    return twod_image

def flatten(xss):
    return [x for xs in xss for x in xs]

def get_pixel(image, row, col, boundrary_behaviour = None):
    if(boundrary_behaviour == "zero"):
        if((row < 0 or row > image["width"])) or (col < 0 or col > image["height"]):
            return 0
        else:
            return image["pixels"][col][row]
    elif(boundrary_behaviour == "extend"):
        if(col <= 0 and row < 0):
            return image["pixels"][0][0]
        elif(col <= 0 and row >= image["width"]):
            return image["pixels"][0][image["width"] - 1] 
        elif(col >= image["height"] and row < 0):
            return image["pixels"][image["height"] - 1][0]
        elif(col >= image["height"] and row < 0):
            return image["pixels"][image["height"] - 1][0]
        elif (col >= image["height"] and row > image["width"]):
            return image["pixels"][image["height"]][image["width"] - 1]
        elif (col == 0 and row > 0):
            return image["pixels"][0][row]
        elif (col > 0 and row == 0):
            return image["pixels"][col][0]
        elif (col > 0 and row == (image["width"] - 1)):
            return image["pixels"][col][image["width"] - 1]
        elif (col == (image["height"] - 1) and row > 0):
            return image["pixels"][image["height"] - 1][row]
        else:
            return image["pixels"][col][row]
    elif (boundrary_behaviour == "wrap"):
        return image["pixels"][col % image["height"]][row  % image["width"]]
    else:
        return image["pixels"][col][row]


def set_pixel(image, row,  col, color):
  
     image["pixels"][col][row] = color


def apply_per_pixel(image, func):
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [[0 for _ in range(image["width"])] for _ in range(image["height"])]
    }
    new_image = oned_to_twod(image)
    for col in range(image["height"]):
        for row in range(image["width"]):
            color = get_pixel(new_image, row, col)
            new_color = func(color)
            set_pixel(result, row, col, new_color)
    result["pixels"] = flatten(result["pixels"])
    return result

def inverted(image):
    return apply_per_pixel(image, lambda color: 255 -color)


# HELPER FUNCTIONS


def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    """
    kernel = [[0,0,0], [0,0,1], [0,0,0]]
    if(boundary_behavior != "zero" and boundary_behavior != "extend" and boundary_behavior != "wrap"):
        return None
    result = {
        "height" : image["height"],
        "width" : image["width"],
        "pixels" : [[0 for _ in range(image["width"])] for _ in range(image["height"])]
    }
    new_image = oned_to_twod(image)
    for col in range(image["height"]):
        for row in range(image["width"]):
            color = get_pixel(new_image, row, col, boundary_behavior)
            for i in range(len(kernel)):
                for j in range(len(i)):
                    color += get_pixel(new_image, row - 1 - i, col - 1 - j, boundary_behavior)
        set_pixel(result, row, col, color)    
    result["pixels"] = flatten(result["pixels"])
    return result


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    raise NotImplementedError


# FILTERS

def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    raise NotImplementedError



# HELPER FUNCTIONS FOR DISPLAYING, LOADING, AND SAVING IMAGES

def print_greyscale_values(image):
    """
    Given a greyscale image dictionary, prints a string representation of the
    image pixel values to the terminal. This function may be helpful for
    manually testing and debugging tiny image examples.

    Note that pixel values that are floats will be rounded to the nearest int.
    """
    out = f"Greyscale image with {image['height']} rows"
    out += f" and {image['width']} columns:\n "
    space_sizes = {}
    space_vals = []

    col = 0
    for pixel in image["pixels"]:
        val = str(round(pixel))
        space_vals.append((col, val))
        space_sizes[col] = max(len(val), space_sizes.get(col, 2))
        if col == image["width"] - 1:
            col = 0
        else:
            col += 1

    for (col, val) in space_vals:
        out += f"{val.center(space_sizes[col])} "
        if col == image["width"]-1:
            out += "\n "
    print(out)


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the "mode" parameter.
    """
    # make folders if they do not exist
    path, _ = os.path.split(filename)
    if path and not os.path.exists(path):
        os.makedirs(path)

    # save image in folder specified (by default the current folder)
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    inv = load_greyscale_image("test_images/bluegill.png")
    inv = inverted(inv)
    save_greyscale_image(inv, "inverted_image.png")
