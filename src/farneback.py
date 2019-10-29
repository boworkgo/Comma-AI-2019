import cv2 as cv
import numpy as np


def get_farneback(name, no_frames):
    if no_frames < 2:
        return

    def get_image(idx):
        return cv.imread("../data/frames/{}{}{}.jpg".format(name, watershed, idx))

    def get_gray(img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    for _ in range(no_frames - 1):
        frame1, frame2 = get_image(_), get_image(_ + 1)
        prv, nxt = get_gray(frame1), get_gray(frame2)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        flow = cv.calcOpticalFlowFarneback(prv, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imwrite("../data/frames/{}_farne_{}.jpg".format(name, _, watershed), img)


get_farneback("train", 20400)
get_farneback("test", 10798)
