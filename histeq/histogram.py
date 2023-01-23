import matplotlib.pyplot as plt
from PIL import Image as pimg
import numpy as np


class HistEqualization:

    @staticmethod
    def get_brightness_values(pic):
        width, height = pic.size
        return [pic.getpixel((x, y)) for x in range(width) for y in range(height)]

    @staticmethod
    def calc_commulative_vals(values, brange):
        commulative = np.zeros((brange,), dtype='int32')
        for val in values:
            commulative[val] += 1

        for i in range(1, brange):
            commulative[i] += commulative[i - 1]

        return commulative

    @staticmethod
    def histogram(pic):
        vals = HistEqualization.get_brightness_values(pic)
        plt.hist(vals, bins=256, range=(0, 255), color='blue', histtype='bar', rwidth=0.8)
        plt.title('Histogram')
        plt.show()

    @staticmethod
    def commulative(pic, brightness_range):
        vals = HistEqualization.get_brightness_values(pic)
        comm = HistEqualization.calc_commulative_vals(vals, brightness_range)

        plt.scatter(np.arange(0, brightness_range), comm, s=3)
        plt.title('Commulative')
        plt.show()


# Read image
image = pimg.open("C:\\Users\\janezs\\Documents\\personal\\distributed-systems\\histeq\\kolesar-neq.jpg")

# Create histogram
HistEqualization.histogram(image)
HistEqualization.commulative(image, 256)