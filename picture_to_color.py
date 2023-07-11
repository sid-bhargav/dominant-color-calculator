from PIL import Image
import binascii
import numpy as np
import scipy

NUM_CLUSTERS = 5

# input: PIL image, output: dominant color
def dominant_color(image):

    # makes an array of all points, so that it's not a 3D array
    array = np.asarray(image)

    shape = array.shape
    array = array.reshape(np.prod(shape[:2]), shape[2]).astype(float)

    mask = np.ones(len(array), dtype=bool)
    for i in range(len(array)):
        if array[i][3] == 0:
            mask[i] = False

    array = array[mask]

    print('clustering')
    codes, dist = scipy.cluster.vq.kmeans(array, NUM_CLUSTERS)

    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(array, codes)      # assign codes
    counts, bins = np.histogram(vecs, len(codes))       # count occurrences

    index_max = np.argmax(counts)                       # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    print('most frequent is %s (#%s)' % (peak, colour))

def main():
    image = Image.open('images\guitar_apple_emoji.png')
    dominant_color(image)


if __name__ == "__main__":
    main()