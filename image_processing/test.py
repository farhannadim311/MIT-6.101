#!/usr/bin/env python3

import os
import pickle
import hashlib

import lab
import pytest

TEST_DIRECTORY = os.path.dirname(__file__)


def object_hash(x):
    return hashlib.sha512(pickle.dumps(x)).hexdigest()


def compare_images(result, expected):
    assert set(result.keys()) == {'height', 'width', 'pixels'}, f'Incorrect keys in dictionary'
    assert result['height'] == expected['height'], 'Heights must match'
    assert result['width'] == expected['width'], 'Widths must match'
    assert len(result['pixels']) == result['height']*result['width'], f"Incorrect number of pixels, expected {result['height']*result['width']}"
    num_incorrect_val = 0
    first_incorrect_val = None
    num_bad_type = 0
    first_bad_type = None
    num_bad_range = 0
    first_bad_range = None

    row, col = 0, 0
    correct_image = True
    for index, (res, exp) in enumerate(zip(result['pixels'], expected['pixels'])):
        if not isinstance(res, int):
            correct_image = False
            num_bad_type += 1
            if not first_bad_type:
                first_bad_type = f'Pixels must all be integers!'
                first_bad_type += f'\nPixel had value {res} at index {index} (row {row}, col {col}).'
        if res < 0 or res > 255:
            num_bad_range += 1
            correct_image = False
            if not first_bad_range:
                first_bad_range = f'Pixels must all be in the range from [0, 255]!'
                first_bad_range += f'\nPixel had value {res} at index {index} (row {row}, col {col}).'
        if res != exp:
            correct_image = False
            num_incorrect_val += 1
            if not first_incorrect_val:
                first_incorrect_val = f'Pixels must match'
                first_incorrect_val += f'\nPixel had value {res} but expected {exp} at index {index} (row {row}, col {col}).'

        if col + 1 == result["width"]:
            col = 0
            row += 1
        else:
            col += 1

    msg = "Image is correct!"
    if first_bad_type:
        msg = first_bad_type + f"\n{num_bad_type} pixel{'s'*int(num_bad_type>1)} had this problem."
    elif first_bad_range:
        msg = first_bad_range + f"\n{num_bad_range} pixel{'s'*int(num_bad_range>1)} had this problem."
    elif first_incorrect_val:
        msg = first_incorrect_val + f"\n{num_incorrect_val} pixel{'s'*int(num_incorrect_val>1)} had incorrect value{'s'*int(num_incorrect_val>1)}."

    assert correct_image, msg


def test_load():
    result = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))
    expected = {
        'height': 11,
        'width': 11,
        'pixels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    compare_images(result, expected)


def test_inverted_1():
    im = lab.load_greyscale_image(os.path.join(TEST_DIRECTORY, 'test_images', 'centered_pixel.png'))
    result = lab.inverted(im)
    expected = {
        'height': 11,
        'width': 11,
        'pixels': [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    }
    compare_images(result, expected)

def test_inverted_2():
    im = {
        "height" : 1,
        "width" : 4,
        "pixels" : [0,74,136,195]
    }
    result = lab.inverted(im)
    expected = {
        "height" : 1,
        "width" : 4,
        "pixels": [255,181,119,60]
    }
    compare_images(result, expected)

@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_inverted_images(fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_invert.png' % fname)
    im = lab.load_greyscale_image(inpfile)
    oim = object_hash(im)
    result = lab.inverted(im)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(im) == oim, 'Be careful not to modify the original image!'
    compare_images(result, expected)


@pytest.mark.parametrize("kernsize", [1, 3, 7])
@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_blurred_images(kernsize, fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_blur_%02d.png' % (fname, kernsize))
    input_img = lab.load_greyscale_image(inpfile)
    input_hash = object_hash(input_img)
    result = lab.blurred(input_img, kernsize)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(input_img) == input_hash, "Be careful not to modify the original image!"
    compare_images(result, expected)

def test_blurred_black_image():
    im = {
        "height": 6,
        "width": 5,
        "pixels": [0] * (6 * 5)
    }

    expected = {
        "height": 6,
        "width": 5,
        "pixels": [0] * (6 * 5)
    }

    # Test with 3x3 blur
    result_3 = lab.blurred(im, 3)
    result_3 = lab.round_and_clip_image(result_3)
    compare_images(result_3, expected)

    # Test with 5x5 blur
    result_5 = lab.blurred(im, 5)
    result_5 = lab.round_and_clip_image(result_5)
    compare_images(result_5, expected)

def test_blurred_centered_pixel():
    im = {
        "height": 11,
        "width": 11,
        "pixels": [0] * (11 * 11)
    }
    # Set center pixel to 255 (row 5, col 5)
    center_index = 5 * 11 + 5
    im["pixels"][center_index] = 255

    # --- Test 3x3 kernel ---
    result_3 = lab.blurred(im, 3)
    result_3 = lab.round_and_clip_image(result_3)

    # Build expected output for 3x3: all 9 pixels around (5,5) = 28
    expected_3 = {
        "height": 11,
        "width": 11,
        "pixels": [0] * (11 * 11)
    }
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            r = 5 + dr
            c = 5 + dc
            idx = r * 11 + c
            expected_3["pixels"][idx] = 28  # 255 / 9 = 28.33 → round to 28

    compare_images(result_3, expected_3)

    # --- Test 5x5 kernel ---
    result_5 = lab.blurred(im, 5)
    result_5 = lab.round_and_clip_image(result_5)

    # Build expected output for 5x5: all 25 pixels around (5,5) = 10
    expected_5 = {
        "height": 11,
        "width": 11,
        "pixels": [0] * (11 * 11)
    }
    for dr in [-2, -1, 0, 1, 2]:
        for dc in [-2, -1, 0, 1, 2]:
            r = 5 + dr
            c = 5 + dc
            idx = r * 11 + c
            expected_5["pixels"][idx] = 10  # 255 / 25 = 10.2 → round to 10

    compare_images(result_5, expected_5)


@pytest.mark.parametrize("kernsize", [1, 3, 9])
@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_sharpened_images(kernsize, fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_sharp_%02d.png' % (fname, kernsize))
    input_img = lab.load_greyscale_image(inpfile)
    input_hash = object_hash(input_img)
    result = lab.sharpened(input_img, kernsize)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(input_img) == input_hash, "Be careful not to modify the original image!"
    compare_images(result, expected)


@pytest.mark.parametrize("fname", ['mushroom', 'twocats', 'chess'])
def test_edges_images(fname):
    inpfile = os.path.join(TEST_DIRECTORY, 'test_images', '%s.png' % fname)
    expfile = os.path.join(TEST_DIRECTORY, 'test_results', '%s_edges.png' % fname)
    input_img = lab.load_greyscale_image(inpfile)
    input_hash = object_hash(input_img)
    result = lab.edges(input_img)
    expected = lab.load_greyscale_image(expfile)
    assert object_hash(input_img) == input_hash, "Be careful not to modify the original image!"
    compare_images(result, expected)
def test_edges_centered_pixel():
    im = {
        "height": 11,
        "width": 11,
        "pixels": [0] * (11 * 11)
    }

    # Set center pixel to 255 (row 5, col 5)
    im["pixels"][5 * 11 + 5] = 255

    expected = {
    "height": 11,
    "width": 11,
    "pixels": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0,
        0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0,
        0, 0, 0, 0, 255, 0,   255, 0, 0, 0, 0,
        0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0,
        0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0
    ]
}

    result = lab.edges(im)
    compare_images(result, expected)
