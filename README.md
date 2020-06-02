## Project Report: __Advanced Lane Line Detection__

<!-- ### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer. I changed something -->

---

**Advanced Lane Finding Project**

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)


[channelInvestigation]: output_images/channelInvestigation.png "Select the best channels"
[three_separate_channels]: output_images/three_separate_channels.png "Three thresholded channels"
[combined_color]: output_images/combined_color.png "Combined color"
[combined_gradient]: output_images/combined_gradient.png "Combine gradient"
[combined_final]: output_images/combined_final.png "Combined final"
[perspective_transform]: output_images/perspective_transform.png "perspective_transform"
[histogram]: output_images/histogram.png "histogram"
[sliding_windows]: output_images/sliding_windows.png "sliding windows"
[search_around_poly]: output_images/search_around_poly.png "search_around_poly"

[test1]: output_images/test_images/test1.jpg "Test 1"
[test2]: output_images/test_images/test2.jpg "Test 2"
[test3]: output_images/test_images/test3.jpg "Test 3"
[test4]: output_images/test_images/test4.jpg "Test 4"
[test5]: output_images/test_images/test5.jpg "Test 5"
[test6]: output_images/test_images/test6.jpg "Test 6"

[image1]: output_images/cameraCalibration.png "Chessboard Camera Calibration"
[image2]: output_images/roadUndistortion.png "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

*Note*: All the helper functions are provided in <a href=https://github.com/chgswin/CarND-Advanced-Lane-Lines/blob/master/advancedCV.py>`advancedCV.py`</a>

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

<!--The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: -->

The first and foremost step is to calibrate the camera to produce undistorted images. To do this, a series of chessboard images taken from different angles are pre-prepared as input. 
These images are then pushed into `cv2.findChessboardCorners()` find the correct corner cordinates of the corners in the chessboard. They, along with an array list of object points (corresponding indices) of the corners of each chessboard, are then used by `cv2.calibrateCamera()` to produce the `distortion matrix`. 
`cv2.undistort()` then use this `distortion matrix` to *undistort* any other images produced by this camera.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

The effect of `cv2.undistort()` could be subtly recognized by change in the shape of the carhood near the bottom of this image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

##### 2.1. Pick the right color channels:

![channel list][channelInvestigation] 

I investigated three common color spaces: Red-Green-Blue, Hue-Saturation-Value, and Hue-Light-Saturation. I then could observe that the lane lines are the clearest to be detected through the __Red__ and __Val__ channel amongst other channels.

To extract the correct lane lines, we need to determine the minimum and maximum thresholds so that all pixels falling in between these two thresholds will be active.

In my notebook, I created a small interactive tool to pick the appropriate thresholds using `ipywidget sliders`.

Color min and max thresholds are:

*    Red = (175, 255)
*    Val = (170, 255)

<!--I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images) -->

![Three thresholded channels][three_separate_channels]

##### 2.2. Pick the right gradient thresholds
I experimented the sobel x gradient, sobel y gradient, magnitude gradient and angle direction gradient. Following the same procedures, I determined the appropriate upper and lower thresholds as follows:

- Sobel operator
    * x direction: [40, 255]
    * y direction: [60, 255]
- Magnitude: [60, 255]
- Direction: [40, 60] (in degree)

##### 2.3. Combine these channels together

![combined color][combined_color] 

For color channels, the __Red__ and __Value__ channels are put together via an __AND__, meaning that the active pixels must be active in both channels.

For gradient channels, the active pixels must satisfy live within direction thresholds and be in one of the following channels: x-sobel, y-sobel or magnitude channel.

![combined gradient][combined_gradient]


The ultimate result should be the combination of pixels that are active in color channel or in gradient channel.

![final][combined_final]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes `warp(image, source, destination)`, which will transform the image, after calculating the transformation matrix based on `source` and `destination` set of points. My source points are hardcoded, where are the vertices of a trapezium where the side edges are parallel along the lane lines. The destination points form a rectangle with 10%. It is noted that the source and destination quadrilaterals are symmetrical. 

```python
source =        [[  190.   720.]
                 [  520.   500.]
                 [  760.   500.]
                 [ 1090.   720.]]
destination =   [[  128.   720.]
                [  128.     0.]
                [ 1152.     0.]
                [ 1152.   720.]]
```
To get the hardcoded source and destination points, it should be useful to have a good reference image. For my project, the `straight_lines1.png` was chosen because of its good symmetry.

![perspective transform][perspective_transform]
<!--The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
-->
#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Following the methods described in the lecture videos, I implemented two methods to detect the lane lines: `find_lane_pixels()`, and `search_around_poly()`.

##### 4.1. Scanning all active pixels with `find_lane_pixels()`
`find_lane_pixels()` scans the whole image for active images. This is done by the __sliding window__ technique. 
* Finding two x cordinates, one for the left lane and another for the right lane where the left lane and right lane mostly appears. That is achieved by calculating two pixel columns with the most active pixels. These two columns will serve as the bases for our search. They are where the two bottommost windows are placed. The search will gradually scan upward, where the centres for each window is situated at the average position of all points within the previous window right below.

With all the points within the sliding windows, the possible lane line is constructed by fitting a quadratic polynomial to these points. 

![sliding windows][sliding_windows]

##### 4.2. Finding the active pixels around the previous curve with `search_around_poly()`
For stable lanes, it is usual for lane lines to be near around previous lane lines. Keeping this in mind, a faster method to scan active pixels only around previous lane lines is constructed. All active pixels within a horizontal offset from previous lines are detected for a new curve to be fitted. This method could effectively speed up the whole process and help reduce possible noise from around the lanes. 

##### 4.3. Further improvement

It is observed that the existing `fit_polynomial` might be prone to errors as it could mistakenly detect other noise lines. To reduce this, an array of weights is used: pixel points near the car (at the bottom of the images) carry more weight than the points far away from the cars. This improvement helps the most when there could be other lines that lie next to the actual lane lines.

![search around previous line][search_around_poly]
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The best fit polynomial provides three coefficients: Ax^2 + Bx + C. The reference formula to calculate the radius of curvature is provided [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php). In the code, radius of curvature for each lane line is provided in the `Line` class. Standalone methods could also be found.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][test1]
![alt text][test2]
![alt text][test3]
![alt text][test4]
![alt text][test5]
![alt text][test6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

[link to my video result](output_videos/project_video.mp4)

[link to my challenge result](output_videos/challenge_video.mp4)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

__Problem:__ The polynomial is fitted but too different from the last frame or it might not be as reasonable to be accepted.

__Proposed Solution:__:
* *Impose a different limit serving as sanity check:* New polynomial is compared against the previous polynomial and the value is only trusted if the coeffients are not too different from the previous one.

* *Implement a average FIR filter:* To ensure that the lane detection is stable, the best fit is the average of the newest found one and four previous polynomials.

* *:*

__Problem:__ Data is not enough to trust the polynomial being fitted

__Proposed Solution:__:

In this case, we might need to delay fitting the new polynomial, using the recent reliable fit and waiting until number of active pixels are adequate to fit a trustful curve.

__Problem:__ The actual lane lines change dramatically and the regression lines do not fit well.

__Proposed Solution__:

* Widen the area of interest horizontally but shorten it vertically. These might help detecting the nearest lane lines more reliable.

### Reference

1. <a href=https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines>Jeremy Shannon's project</a>
2. <a href=https://github.com/tranlyvu/Self-Driving-Car-Engineer-Part-1/tree/master/Advanced%20Lane%20Lines>Tran Ly Vu's project</a>
