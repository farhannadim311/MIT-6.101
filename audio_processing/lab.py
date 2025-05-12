"""
6.101 Lab:
Audio Processing
"""

import wave
import struct

# No additional imports allowed!


def backwards(sound):
    """
    Returns a new sound containing the samples of the original in reverse
    order, without modifying the input sound.

    Args:
        sound: a dictionary representing the original mono sound

    Returns:
        A new mono sound dictionary with the samples in reversed order
    """
    rev_index = len(sound["samples"]) - 1
    rev_sound = {}
    rev_sound["rate"] = sound["rate"]
    rev_sound["samples"] = [0] * (rev_index + 1)
    for i in range(len(sound["samples"])):
        rev_sound["samples"][i] = sound["samples"][rev_index]
        rev_index -= 1

    return rev_sound


def mix(sound1, sound2, p):
    """
    Returns a mix of the two sounds passed into the parameter

    Args:
        sound1: a dictionary representing a sound
        sound2: a dictionary representing another sound
        p: a mixing parameter which is used to multiply to get the new sound
    Returns:
        A new mono sound dictionary with the two sounds mixed together
    """
    # mix 2 good sounds
    if (
        ("rate" in sound1) is False
        or ("rate" in sound2) is False
        or (sound1["rate"] == sound2["rate"]) is False
    ):

        print("no")
        return None

    r = sound1["rate"]  # get rate
    sound1 = sound1["samples"]
    sound2 = sound2["samples"]
    if len(sound1) < len(sound2):
        max_len = len(sound2)
    elif len(sound2) < len(sound1):
        max_len = len(sound1)
    else:
        max_len = len(sound2)
    sound = []
    x = 0
    while x <= max_len:
        if x < len(sound1) and x < len(sound2):
            sound.append(p * sound1[x] + sound2[x] * (1 - p))
        elif len(sound2) <= x < len(sound1):
            sound.append(p * sound1[x])
        else:
            sound.append(sound2[x] * (1 - p))
        x += 1
        if x == max_len:  # end
            break

    return {"rate": r, "samples": sound}  # return new sound


def convolve(sound, kernel):
    """
    Compute a new sound by convolving the given input sound with the given
    kernel.  Does not modify input sound.

    Args:
        sound: a dictionary representing the original mono sound
        kernel: list of numbers, the signal with which the sound should be
                convolved

    Returns:
        A new mono sound dictionary resulting from the convolution.
    """
    new_sound = []
    res = {}
    res["rate"] = sound["rate"]
    res["samples"] = [0] * (len(sound) + (len(kernel) ))
    for row in range(len(kernel)):
        inner_list = []
        for col in range(len(sound) + (len(kernel))):
            if(col > len(sound)):
                inner_list.append(0)
            else:
                inner_list.append(sound["samples"][col])
        new_sound.append(inner_list)
    for i in range(len(new_sound)):
        for j in range(len(new_sound[i])):
            if(kernel[j] != 0):
                new_sound[i][j] *= kernel[j]
            if(kernel[j] != 0):
                if(i + j > len(new_sound[i])):
                    for k in range(j + i):
                        new_sound[i][j] = 0
                else:
                    new_sound[i][i + j] = new_sound[i][j]
        if (j == len(kernel) - 1):
            break
    for i in new_sound:
        for idx, j in enumerate(new_sound[i]):
            res["samples"][idx] += j
    return res



def echo(sound, num_echoes, delay, scale):
    """
    Compute a new sound consisting of several scaled-down and delayed versions
    of the input sound. Does not modify input sound.

    Args:
        sound: a dictionary representing the original mono sound
        num_echoes: int, the number of additional copies of the sound to add
        delay: float, the amount of seconds each echo should be delayed
        scale: float, the amount by which each echo's samples should be scaled

    Returns:
        A new mono sound dictionary resulting from applying the echo effect.
    """
    raise NotImplementedError


def pan(sound):
    raise NotImplementedError


def remove_vocals(sound):
    raise NotImplementedError


def bass_boost_kernel(n_val, scale=0):
    """
    Construct a kernel that acts as a bass-boost filter.

    We start by making a low-pass filter, whose frequency response is given by
    (1/2 + 1/2cos(Omega)) ^ n_val

    Then we scale that piece up and add a copy of the original signal back in.
    """
    # make this a fake "sound" so that we can use the convolve function
    base = {"rate": 0, "samples": [0.25, 0.5, 0.25]}
    kernel = {"rate": 0, "samples": [0.25, 0.5, 0.25]}
    for i in range(n_val):
        kernel = convolve(kernel, base["samples"])
    kernel = kernel["samples"]

    # at this point, the kernel will be acting as a low-pass filter, so we
    # scale up the values by the given scale, and add in a value in the middle
    # to get a (delayed) copy of the original
    kernel = [i * scale for i in kernel]
    kernel[len(kernel) // 2] += 1

    return kernel


# below are helper functions for converting back-and-forth between WAV files
# and our internal dictionary representation for sounds


def load_wav(filename, stereo=False):
    """
    Given the filename of a WAV file, load the data from that file and return a
    Python dictionary representing that sound
    """
    file = wave.open(filename, "r")
    chan, bd, sr, count, _, _ = file.getparams()

    assert bd == 2, "only 16-bit WAV files are supported"

    out = {"rate": sr}

    if stereo:
        left = []
        right = []
        for i in range(count):
            frame = file.readframes(1)
            if chan == 2:
                left.append(struct.unpack("<h", frame[:2])[0])
                right.append(struct.unpack("<h", frame[2:])[0])
            else:
                datum = struct.unpack("<h", frame)[0]
                left.append(datum)
                right.append(datum)

        out["left"] = [i / (2**15) for i in left]
        out["right"] = [i / (2**15) for i in right]
    else:
        samples = []
        for i in range(count):
            frame = file.readframes(1)
            if chan == 2:
                left = struct.unpack("<h", frame[:2])[0]
                right = struct.unpack("<h", frame[2:])[0]
                samples.append((left + right) / 2)
            else:
                datum = struct.unpack("<h", frame)[0]
                samples.append(datum)

        out["samples"] = [i / (2**15) for i in samples]

    return out


def write_wav(sound, filename):
    """
    Given a dictionary representing a sound, and a filename, convert the given
    sound into WAV format and save it as a file with the given filename (which
    can then be opened by most audio players)
    """
    outfile = wave.open(filename, "w")

    if "samples" in sound:
        # mono file
        outfile.setparams((1, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = [int(max(-1, min(1, v)) * (2**15 - 1)) for v in sound["samples"]]
    else:
        # stereo
        outfile.setparams((2, 2, sound["rate"], 0, "NONE", "not compressed"))
        out = []
        for left, right in zip(sound["left"], sound["right"]):
            left = int(max(-1, min(1, left)) * (2**15 - 1))
            right = int(max(-1, min(1, right)) * (2**15 - 1))
            out.append(left)
            out.append(right)

    outfile.writeframes(b"".join(struct.pack("<h", frame) for frame in out))
    outfile.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place to put your
    # code for generating and saving sounds, or any other code you write for
    # testing, etc.

    # here is an example of loading a file (note that this is specified as
    # sounds/hello.wav, rather than just as hello.wav, to account for the
    # sound files being in a different directory than this file)
    #synth = load_wav("sounds/synth.wav")
    #water = load_wav("sounds/water.wav")
    #write_wav(mix(synth, water, 0.2), "mix.wav")
    pass
