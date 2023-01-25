import matplotlib.pyplot as plt
from PIL import Image as pimg
import numpy as np


class HistEqualization:

    @staticmethod
    def get_brightness_values(pic):
        width, height = pic.size
        pic = pic.convert("RGB")
        return [sum(list(pic.getpixel((x, y))))//3 for x in range(width) for y in range(height)]

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


def draw_hist_and_commulative(subfolder, image)
    # Read image
    image = pimg.open(f"C:\\Users\\janezs\\Documents\\personal\\distributed-systems\\histeq\\{subfolder}\\{image}.jpg")

    # Create histogram
    HistEqualization.histogram(image)
    HistEqualization.commulative(image, 256)

def analysis():
    data = [
        ("500",         500 ** 2)
        ("640",         640 ** 2)
        ("800",         800 ** 2)
        ("1000",        1000 ** 2)
        ("1024_640",    1024 * 640)
        ("1200",        1200 * 2)
        ("1680_1050",   1680 * 1050)
        ("1920_1080",   1920 * 1080)
        ("2560_1080",   2560 * 1080)
        ("2560_1440",   2560 * 1440)
        ("4k",          3840 * 2160)
        ("4kp",         4056 * 3040)
    ]

    names = [name for name, _ in data]
    resolutions = [res for _, res in data]


#draw_hist_and_commulative('out_cpu', '500')
#analysis
