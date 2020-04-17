import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_folder = "datasets/tdt4265/original_1280x960/"

if __name__ == "__main__":
    files = os.listdir(image_folder)
    images = []
    for file in files:
        path = os.path.join(image_folder, file)
        img = cv2.imread(path)
        edit = cv2.resize(img, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)
        images.append((img, edit))
    for orig, edit in images:
        fig = plt.figure()
        sub = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(orig)
        sub.set_title("Original")
        sub = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(edit)
        sub.set_title("Edit")
        plt.show()
