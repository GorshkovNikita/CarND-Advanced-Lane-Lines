## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistorted_image.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_example.png "Binary Example"
[image4]: ./examples/warped_with_area.png "Warp Example"
[image5]: ./examples/fitted_polynomial.png "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./examples/distorted_image.png "Distorted"
[image8]: ./examples/road_undistorted.png "Road undistorted"
[image9]: ./examples/source_image_with_area.png "Source image with area"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 7 through 23 of the file called `calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image7]

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Here is undistorted version of this image:

![alt text][image8]

The code for this step is contained in line 20 of the file called `main.py`. 
As you can see I use `mtx` (camera matrix) and `dst` (distortion coefficients) variables, which was obtained previously on calibration step.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

For this step I used a combination of color and gradient thresholds to generate a binary image. The code is contained
in the file called `thresholding.py`. I decided to use thresholding for S axis in HLS color space, R axis in RGB color space and for X axis in
Sobel-filtered image, because this combination detect lane line very clear.
Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
The code for my perspective transform includes a function called `warp()`, which appears in lines 18 through 19, 
a function called `warp_and_sharpen()` and a funcation called `rotation_matrix()` in the file `warper.py`. 
The `warp()` function takes as inputs an image (`img`), as well as rotation matrix. After warping a binary image 
becomes blurred and I use `warp_and_sharpen()` to sharpen this binary image. Function `rotation_martix()` is used for obtaining 
rotation matrix based on source and destination points.
I chose the hardcode the source and destination points in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 210, 718      | 250, 720      | 
| 601, 448      | 250, 0        |
| 680, 446      | 1000, 0       |
| 1106, 718     | 1000, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image 
and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I identified lane-line pixels in function `find_line_pixels()` in file `line_fitting.py`. Firstly, I represented image
as `nwidnows` windows. In each window I found positions with maximum neighbouring pixels. For this I used `np.convolve()`.
For each new window I used previously found positions as starting point.
Then I fitted found pixels to the polynomial of second degree in file `line_fitting.py` with function `fit_lines()`,
where I casted pixels to meters and, then, computed polynomial coefficients.

Here is the result I obtained:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature in lines 22 through 28 in my code in file `line_fitting.py`.
This function takes as arguments left and right line pixels. Then I cast pixels to meters and compute line fit using 
`cv2.polyfit()` function. After that, I use the formula for computing radius.
I calculated the position of vehicle in respect to center in lines 50 through 54 in file `line_fitting.py`, where
variable `warped` represents image with plotted lines on it. So I can find positions of these lines. Then I compute
real position of the car (which is center of image), and ideal position (which is center between two lines). The 
difference between these values is the offset from center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 31 through 58 in my code in `line_fitting.py` in the function `plot_lines()`. 
For that I needed to filter out coordinates, which fall out of image after computing line coordinates. After that
I created image only with lane-lines. Then I warped this image back to initial coordinates of source image. And finally
overlaid this image on source image.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem in this project was shadows on the road. To tackle this problem I decided to use RGB color space in
thresholding stage of pipeline to filter out all dark pixels.

Obviously, this pipeline will fail whenever it's hard to detect lines even for human (for example, when road is covered 
by snow). In such case, we could use surrounding cars to detect lanes.
