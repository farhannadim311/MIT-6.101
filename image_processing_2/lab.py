#!/usr/bin/env python3

"""
6.101 Lab:
Image Processing 2
"""

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
import os
# import typing  # optional import
from PIL import Image 

# COPY THE FUNCTIONS THAT YOU IMPLEMENTED IN IMAGE PROCESSING PART 1 BELOW!

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
        if((row < 0 or row > image["height"] - 1)) or (col < 0 or col > image["width"] - 1):
            return 0
        else:
            return image["pixels"][row][col]
    elif(boundrary_behaviour == "extend"):
        if ((row >= 0 and row <= image["height"] -1) and (col >= 0 and col <= image["width"] - 1)):
            return image["pixels"][row][col]
        elif (row <= 0 and col <= 0):
            return image["pixels"][0][0]
        elif (row <= 0 and (col >= image["width"] -1)):
            return image["pixels"][0][image["width"] - 1]
        elif (row <= 0 and (col > 0 and col <= image["width"] - 1)):
            return image["pixels"][0][col]
        elif (row >= image["height"] - 1 and col <= 0):
            return image["pixels"][image["height"] -1][0]
        elif ((row > 0 and row <= image["height"] - 1) and col <= 0 ):
            return image["pixels"][row][0]
        elif ((row >= image["height"] - 1) and col >= image["width"] - 1):
            return image["pixels"][image["height"] - 1][image["width"] -1]
        elif ((row >= image["height"] - 1) and (col > 0 and col <= image["width"] -1 )):
              return image["pixels"][image["height"] - 1][col]
        elif ((row > 0 and row <= image["height"] - 1) and col >= image["width"] - 1):
            return image["pixels"][row][image["width"] - 1]
    elif (boundrary_behaviour == "wrap"):
        return image["pixels"][row % image["height"]][col  % image["width"]]
    return image["pixels"][row][col]


def set_pixel(image, row,  col, color):
  
     image["pixels"][row][col] = color


def apply_per_pixel(image, func):
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [[0 for _ in range(image["width"])] for _ in range(image["height"])]
    }
    new_image = oned_to_twod(image)
    for row in range(image["height"]):
        for col in range(image["width"]):
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
    Apply correlation between a 2D kernel and the image.
    Supports 'zero', 'extend', and 'wrap' boundary behavior.
    Returns a new image dictionary with floating-point pixel values.
    """
    if boundary_behavior not in ("zero", "extend", "wrap"):
        return None

    # Prepare output image
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [[0 for _ in range(image["width"])] for _ in range(image["height"])]
    }

    # Convert 1D pixel list to 2D for easier access
    image_2d = oned_to_twod(image)

    # Get kernel dimensions and center offset
    k_height = len(kernel)
    k_width = len(kernel[0])
    k_center_row = k_height // 2
    k_center_col = k_width // 2

    # Loop over each image pixel
    for row in range(image["height"]):
        for col in range(image["width"]):
            acc = 0  # accumulator for weighted sum
            for kr in range(k_height):
                for kc in range(k_width):
                    image_r = row - k_center_row + kr
                    image_c = col - k_center_col + kc
                    pixel = get_pixel(image_2d, image_r, image_c, boundary_behavior)
                    acc += pixel * kernel[kr][kc]
            result["pixels"][row][col] = acc

    # Flatten pixel list before returning
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
    for idx, val in enumerate(image["pixels"]):
        val = round(val)             
        if val < 0:
            val = 0
        elif val > 255:
            val = 255
        image["pixels"][idx] = val
    return image

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
    kernel = blur_kernel(kernel_size)
    boundary = "extend"
    result = correlate(image, kernel, boundary)
    result = round_and_clip_image(result)
    return result

def blur_kernel(size):
    val = 1 / (size * size)
    array = [[val for _ in range(size)] for _ in range(size)]
    return array

def sharpened(image, kernel_size):
    """
    Applies an unsharp mask to the given greyscale image to produce a sharpened version.

    This operation enhances edges and fine details by subtracting a blurred version
    of the image from the original image and scaling the difference. Mathematically,
    the sharpened image S is computed as:

        S = 2 * Original - Blurred

    where the blurred image is created using a box blur of the specified kernel size.

    The result is then rounded and clipped to ensure all pixel values are valid 
    integers in the range [0, 255].

    Parameters:
        image (dict): A 6.101-format greyscale image dictionary with 'height', 
                      'width', and 'pixels' keys.
        kernel_size (int): Size of the box blur kernel to use for the unsharp mask.

    Returns:
        dict: A new 6.101-format image dictionary representing the sharpened image.
    """
    kernel = blur_kernel(kernel_size)
    boundary = "extend"
    blur = correlate(image, kernel, boundary)
    result = {
        "height" : image["height"],
        "width" : image["width"],
        "pixels" : [[0 for _ in range(image["width"])] for _ in range(image["height"])]
    }
    twod_image = oned_to_twod(image)
    two_blur = oned_to_twod(blur)
    for i in range(image["height"]):
        for j in range(image["width"]):
            result["pixels"][i][j] = 2 * twod_image["pixels"][i][j] - two_blur["pixels"][i][j]
    result["pixels"] = flatten(result["pixels"])
    result = round_and_clip_image(result)
    return result

def edges(image):
    """
    Applies the Sobel edge detection filter to a greyscale image.

    This operation detects edges by combining the results of two separate 
    correlations: one using the Sobel kernel for detecting horizontal changes (Gx),
    and one for vertical changes (Gy). These are defined as:

        Gx = [[-1, -2, -1],
              [ 0,  0,  0],
              [ 1,  2,  1]]

        Gy = [[-1,  0,  1],
              [-2,  0,  2],
              [-1,  0,  1]]

    After applying each kernel to the image (with 'extend' boundary behavior),
    the output pixel at position (r, c) is computed as:

        sqrt(Gx[r, c]^2 + Gy[r, c]^2)

    This highlights areas in the image where pixel values change quickly,
    which typically correspond to edges.

    The final output is rounded and clipped so that all pixel values are integers 
    within the range [0, 255].

    Parameters:
        image (dict): A 6.101-format greyscale image dictionary with 'height',
                      'width', and 'pixels' keys.

    Returns:
        dict: A new 6.101-format image dictionary representing the edge-detected image.
    """
    Gx = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1] ]
    Gy = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]
    two_d = oned_to_twod(image)
    boundary = "extend"
    result = {
        "height" : image["height"],
        "width" : image["width"],
        "pixels" : [[0 for _ in range(image["width"])] for _ in range(image["height"])]
    }
    temp1 = correlate(image, Gx, boundary)
    temp2 = correlate(image, Gy, boundary)
    temp1_two = oned_to_twod(temp1)
    temp2_two = oned_to_twod(temp2)
    for i in range(image["height"]):
        for j in range(image["width"]):
            result["pixels"][i][j] = math.sqrt((temp1_two["pixels"][i][j])**2 + (temp2_two["pixels"][i][j])**2)
    result["pixels"] = flatten(result["pixels"])
    result = round_and_clip_image(result)
    return result

# VARIOUS FILTERS


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def color(image):
        # Extract RGB channels
        red = {
            "height": image["height"],
            "width": image["width"],
            "pixels": [pixel[0] for pixel in image["pixels"]]
        }
        green = {
            "height": image["height"],
            "width": image["width"],
            "pixels": [pixel[1] for pixel in image["pixels"]]
        }
        blue = {
            "height": image["height"],
            "width": image["width"],
            "pixels": [pixel[2] for pixel in image["pixels"]]
        }
        
        # Apply filter to each channel
        red_filtered = filt(red)
        green_filtered = filt(green)
        blue_filtered = filt(blue)
        
        # Recombine channels
        result = {
            "height": image["height"],
            "width": image["width"],
            "pixels": [
                (red_filtered["pixels"][i], green_filtered["pixels"][i], blue_filtered["pixels"][i])
                for i in range(len(red_filtered["pixels"]))
            ]
        }
        return result
    return color


def make_blur_filter(kernel_size):
    """
    Returns a blur filter function with the specified kernel size.
    """
    def blur_filter(image):
        return blurred(image, kernel_size)
    return blur_filter


def make_sharpen_filter(kernel_size):
    """
    Returns a sharpen filter function with the specified kernel size.
    """
    def sharpen_filter(image):
        return sharpened(image, kernel_size)
    return sharpen_filter


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def cascaded_filter(image):
        result = image
        for filt in filters:
            result = filt(result)
        return result
    return cascaded_filter


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    result = image.copy()
    result["pixels"] = image["pixels"][:]  # Make a copy of pixels list
    
    for _ in range(ncols):
        # Convert to greyscale
        grey = greyscale_image_from_color_image(result)
        
        # Compute energy
        energy = compute_energy(grey)
        
        # Compute cumulative energy map
        cem = cumulative_energy_map(energy)
        
        # Find minimum energy seam
        seam = minimum_energy_seam(cem)
        
        # Remove the seam
        result = image_without_seam(result, seam)
    
    return result


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.
    Uses the standard luminosity formula: 0.299*R + 0.587*G + 0.114*B

    Returns a greyscale image (represented as a dictionary).
    """
    grey_pixels = [
        round(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
        for pixel in image["pixels"]
    ]
    return {
        "height": image["height"],
        "width": image["width"],
        "pixels": grey_pixels
    }


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function) greyscale image, computes a "cumulative energy map" as described
    in the lab 2 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    height = energy["height"]
    width = energy["width"]
    
    # Convert to 2D for easier manipulation
    energy_2d = oned_to_twod(energy)
    
    # Initialize cumulative energy map
    cem = {
        "height": height,
        "width": width,
        "pixels": [[0 for _ in range(width)] for _ in range(height)]
    }
    
    # First row is just the energy values
    for col in range(width):
        cem["pixels"][0][col] = energy_2d["pixels"][0][col]
    
    # Fill in remaining rows using dynamic programming
    for row in range(1, height):
        for col in range(width):
            # Get minimum of three adjacent pixels from previous row
            min_prev = cem["pixels"][row - 1][col]
            
            if col > 0:
                min_prev = min(min_prev, cem["pixels"][row - 1][col - 1])
            if col < width - 1:
                min_prev = min(min_prev, cem["pixels"][row - 1][col + 1])
            
            cem["pixels"][row][col] = energy_2d["pixels"][row][col] + min_prev
    
    # Flatten to 1D
    cem["pixels"] = flatten(cem["pixels"])
    return cem


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map dictionary, returns a list of the indices into
    the 'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    height = cem["height"]
    width = cem["width"]
    
    # Convert to 2D
    cem_2d = oned_to_twod(cem)
    
    # Find minimum value in bottom row
    bottom_row = cem_2d["pixels"][height - 1]
    min_col = bottom_row.index(min(bottom_row))
    
    # Trace back up to find the seam
    seam = []
    current_col = min_col
    
    for row in range(height - 1, -1, -1):
        # Add current position to seam (as 1D index)
        seam.append(row * width + current_col)
        
        if row > 0:
            # Find minimum adjacent pixel in previous row
            min_energy = cem_2d["pixels"][row - 1][current_col]
            next_col = current_col
            
            # Check left
            if current_col > 0 and cem_2d["pixels"][row - 1][current_col - 1] < min_energy:
                min_energy = cem_2d["pixels"][row - 1][current_col - 1]
                next_col = current_col - 1
            
            # Check right
            if current_col < width - 1 and cem_2d["pixels"][row - 1][current_col + 1] < min_energy:
                min_energy = cem_2d["pixels"][row - 1][current_col + 1]
                next_col = current_col + 1
            
            current_col = next_col
    
    return seam


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    seam_set = set(seam)
    new_pixels = [
        pixel for idx, pixel in enumerate(image["pixels"])
        if idx not in seam_set
    ]
    
    return {
        "height": image["height"],
        "width": image["width"] - 1,  # One column removed
        "pixels": new_pixels
    }


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


def print_color_values(image):
    """
    Given a color image dictionary, prints a string representation of the
    image pixel values to the terminal. This function may be helpful for
    manually testing and debugging tiny image examples.

    Note that RGB values will be rounded to the nearest int.
    """
    out = f"Color image with {image['height']} rows"
    out += f" and {image['width']} columns:\n"
    space_sizes = {}
    space_vals = []

    col = 0
    for pixel in image["pixels"]:
        for color in range(3):
            val = str(round(pixel[color]))
            space_vals.append((col, color, val))
            space_sizes[(col, color)] = max(len(val), space_sizes.get((col, color), 0))
        if col == image["width"] - 1:
            col = 0
        else:
            col += 1

    for (col, color, val) in space_vals:
        space_val = val.center(space_sizes[(col, color)])
        if color == 0:
            out += f" ({space_val}"
        elif color == 1:
            out += f" {space_val} "
        else:
            out += f"{space_val})"
        if col == image["width"]-1 and color == 2:
            out += "\n"
    print(out)


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    # make folders if they do not exist
    path, _ = os.path.split(filename)
    if path and not os.path.exists(path):
        os.makedirs(path)

    # save image in folder specified (by default the current folder)
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
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
    by the 'mode' parameter.
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
    pass
