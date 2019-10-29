import cv2 as cv
import numpy as np


def make_frames(name, no_frames):
    cap = cv.VideoCapture("../data/{}.mp4".format(name))
    for _ in range(no_frames):
        ret, frame = cap.read()
        frame = frame[220:, :]  # Crop out sky
        frame = cv.convertScaleAbs(frame, alpha=1.5, beta=25)  # Brighten
        frame = cv.bilateralFilter(frame, 9, 75, 75)  # Reduce noise
        frame = cv.resize(frame, (128, 128))  # Resize for CNN
        cv.imwrite("../data/frames/{}{}.jpg".format(name, _), frame)
    cap.release()


def make_frames_watershed(name, no_frames):
    cap = cv.VideoCapture("../data/{}.mp4".format(name))
    for _ in range(no_frames):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        ret, markers = cv.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv.watershed(frame, markers)
        frame[markers == -1] = [255, 0, 0]
        frame = cv.resize(frame, (128, 128))
        cv.imwrite("../data/frames/{}_watershed_{}.jpg".format(name, _), frame)
    cap.release()


# make_frames("train", 20400)
# make_frames("test", 10798)
make_frames_watershed("train", 20400)
make_frames_watershed("test", 10798)
