## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

## HOW TO USE

### 1. Camera Calibration

```
python src/camera.py --cal_img_dir /path/to/calibration/chessboard/images --calibration_output_dir /path/to/directory/to/store/calibrated/images
```

more arguments
```
--input_img input image to undistort
--output_dir directory to save undistorted image
--cfg_filename camera configuration file name

```

### 2. Perspective Transform

After calibrating the camera, we need to get the perspective transform of the road. This script helps us getting the transformation , and inverse transformation matrix.

```
python perspective_transform.py --image /path/to/image --cfg_filename /path/to/configuratio/file/to/save/configuration --camera_cfg_filename /path/to/camera/configuration/file
```

more arguments:

```
threshold, threshold for perspective transformation. default: 400
plot, to plot or not to plot.

```
