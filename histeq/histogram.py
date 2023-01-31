import matplotlib.pyplot as plt
from PIL import Image as pimg
import numpy as np
from math import log

class HistEqualization:

    @staticmethod
    def get_brightness_values(pic):
        width, height = pic.size
        pic = pic.convert("RGB")
        return [sum(list(pic.getpixel((x, y)))) // 3 for x in range(width) for y in range(height)]

    @staticmethod
    def calc_commulative_vals(values, brange):
        commulative = np.zeros((brange,), dtype='int32')
        for val in values:
            commulative[val] += 1

        for i in range(1, brange):
            commulative[i] += commulative[i - 1]

        return commulative

    @staticmethod
    def histogram(pic, name):
        plt.clf()
        vals = HistEqualization.get_brightness_values(pic)
        plt.hist(vals, bins=256, range=(0, 255), color='blue', histtype='bar', rwidth=0.8)
        plt.title('Histogram')
        # plt.show()
        plt.savefig(f'C:\\Users\\janezs\\Desktop\\vpsa\\results\\{name}o.jpg')

    @staticmethod
    def commulative(pic, brightness_range, name):
        plt.clf()
        vals = HistEqualization.get_brightness_values(pic)
        comm = HistEqualization.calc_commulative_vals(vals, brightness_range)

        plt.scatter(np.arange(0, brightness_range), comm, s=3)
        plt.title('Commulative')
        # plt.show()
        plt.savefig(f'C:\\Users\\janezs\\Desktop\\vpsa\\results\\{name}o2.jpg')

    @staticmethod
    def calc_avg(filename):
        total, count = 0, 0

        with open(filename, 'r') as f:
            for line in f:
                try:
                    total += float(line)
                    count += 1
                except ValueError:
                    pass

        return total / count if count > 0 else 0

    @staticmethod
    def read_chunks(filename):
        data = []
        chunk_size = 5

        with open(filename, 'r') as f:
            for line in f:
                try:
                    data.append(float(line.split(" ")[1]))
                except ValueError:
                    pass

        chunks = [tuple(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)]
        return chunks



    @staticmethod
    def make_graphs():
        for img in ("500", "640", "800", "1000", "1024_640", "1200", "1680_1050", "1920_1080", "2560_1080", "2560_1440", "4k", "4kp"):
            # Read image
            image = pimg.open(f"C:\\Users\\janezs\\Desktop\\vpsa\\results\\{img}.jpg")

            # Create histogram
            HistEqualization.histogram(image, img)
            HistEqualization.commulative(image, 256, img)


    @staticmethod
    def step_analysis():
        avgs = []
        imgs = np.array(("500", "1024_640", "1920_1080", "4k"))

        for img in imgs:
            chunks = HistEqualization.read_chunks(f"C:\\Users\\janezs\\Desktop\\vpsa\\times\\chunks\\{img}.txt")
            first = np.average(np.array([step[1] for step in chunks]))
            second = np.average(np.array([step[2] for step in chunks]))
            third = np.average(np.array([step[3] for step in chunks]))
            fourth = np.average(np.array([step[4] for step in chunks]))
            avgs.append(np.array((first, second, third, fourth)))

        x = np.arange(4)
        width = 0.2

        plt.bar(avg[0] - 0.2, first, width, color='cyan')
        plt.bar(avg[1], second, width, color='orange')
        plt.bar(avg[2] + 0.2, third, width, color='green')
        plt.bar(avg[3] + 0.4, fourth, width, color='red')
        plt.legend(list(imgs))
        plt.xlabel('Steps')
        plt.ylabel('Time (milliseconds)')
        plt.xticks(x, ['Histogram', 'CDF', 'Min', 'Equalize'])

        #plt.show()
        plt.savefig(f'C:\\Users\\janezs\\Desktop\\vpsa\\results\\chunks.jpg')



    @staticmethod
    def analysis():
        data = [
            ("500", 500 ** 2),
            ("640", 640 ** 2),
            ("800", 800 ** 2),
            ("1024_640", 1024 * 640),
            ("1000", 1000 ** 2),
            ("1200", 1200 ** 2),
            ("1680_1050", 1680 * 1050),
            ("1920_1080", 1920 * 1080),
            ("2560_1080", 2560 * 1080),
            ("2560_1440", 2560 * 1440),
            ("4k", 3840 * 2160),
            ("4kp", 4056 * 3040)
        ]

        names = [name for name, _ in data]
        resolutions = [res for _, res in data]
        cpu_avgs = [HistEqualization.calc_avg(f"C:\\Users\\janezs\\Desktop\\vpsa\\times\\cpu\\{name}.txt") for name in names]
        cuda_avgs = [HistEqualization.calc_avg(f"C:\\Users\\janezs\\Desktop\\vpsa\\times\\cuda\\{name}.txt") for name in names]

        cpu_avgs_log = [log(val, 2) for val in cpu_avgs]
        cuda_avgs_log = [log(val, 2) for val in cuda_avgs]

        per_10million_pixels = [res / 10_000_000 for res in resolutions]
        cpu_avgs_normalized = [val/per_10million_pixels[i] for i, val in enumerate(cpu_avgs)]
        cuda_avgs_normalized = [val/per_10million_pixels[i] for i, val in enumerate(cuda_avgs)]

        cpu_avgs_normalized_log = [log(val, 2) for val in cpu_avgs_normalized]
        cuda_avgs_normalized_log = [log(val, 2) for val in cuda_avgs_normalized]
        plt.rcParams["figure.figsize"] = (15, 7)

        # first basic analysis
        plt.clf()
        plt.plot(names, cpu_avgs, label='CPU')
        plt.plot(names, cuda_avgs, label='CUDA')
        plt.xlabel('Images')
        plt.ylabel('Performance')
        plt.title('Performance comparison between images')
        plt.xticks(rotation=25)
        plt.legend()
        plt.show()

        # basic analysis with log applied
        plt.clf()
        plt.plot(names, cpu_avgs_log, label='CPU')
        plt.plot(names, cuda_avgs_log, label='CUDA')
        plt.xlabel('Images')
        plt.ylabel('log(Performance)')
        plt.title('Performance comparison between images - with log')
        plt.xticks(rotation=25)
        plt.legend()
        plt.show()

        # normalized analysis
        plt.clf()
        plt.plot(per_10million_pixels, cpu_avgs_normalized, label='CPU')
        plt.plot(per_10million_pixels, cuda_avgs_normalized, label='CUDA')
        plt.xlabel('10 million pixels')
        plt.ylabel('Performance')
        plt.title('Performance comparison based on amount of pixels')
        plt.legend()
        plt.show()

        # normalized analysis with log applied
        plt.clf()
        plt.plot(per_10million_pixels, cpu_avgs_normalized_log, label='CPU')
        plt.plot(per_10million_pixels, cuda_avgs_normalized_log, label='CUDA')
        plt.xlabel('10 million pixels')
        plt.ylabel('log(Performance)')
        plt.title('Performance comparison based on amount of pixels - with log')
        plt.ylim((0, 15))
        plt.legend()
        plt.show()

HistEqualization.step_analysis()

