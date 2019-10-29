### Summary

To predict the speed of a car given dashcam footage, I used Farneback's algorithm to measure pixel movements and fed the resulting images to a CNN.

### Resources

- [Farneback code I copied from](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html)
- [CNN Tutorial I followed](https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/)
- [CNN - Very clear explanation](https://www.aiworkbox.com/lessons/how-to-define-a-convolutional-layer-in-pytorch)
- [NVIDIA CNN Model I based mine's off of](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
- [Watershed code I copied from](https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html)

### Todo

- Image Augmentation (imgaug)
- Watershedding (Segment images)

### Running

``` 
python3 frames.py
python3 farneback.py
python3 train.py
python3 test.py
python3 plot.py
```
