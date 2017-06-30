
Advanced Lane Finding Project
---

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

[image2]: ./examples/undistortion.jpg "Undistortion"
[image3]: ./examples/thresholding.jpg "Thresholding"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.png "Output"
[video1]: ./project_video_processed.mp4 "Video"
[video2]: ./challenge_video_processed.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in `get_mtx_dist` function. It reads the calibration images and calculate the calibration matrix and distortion coefficients. It needs only to be run once and data can be saved and later loaded when needed, as in line #455. I basically followed the same method in the video lectures to calculate both parameters.

Then in the very begining of `process_image` function I undistort a given image using the data loaded.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the calibration matrix and distortion coefficients above, here are the images before and after undistortion.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I experimented with many combinations of thresholding including RGB, HLS, HSV and Sobel thresholding. The approach I take is to use `test_threshold` function to read sample images and output thresholded binary files.

After lots of experiments, I ended using the formula `HSV | S(hls) | RGB`. HSV is to detect yellow lines and RGB is to detect white lines and S channel is a complimentary. The reason I didn't use any Sobel thresholding is I find it poorly especially when lines are darker (e.g. under bridge) and include lots of noise (e.g. black tar line).

This image include the result of applying different thresholding. `color_binary` is the one used in the project.

`color_threshold` is the function used, which internally calls `hsv_threshold`, `rgb_threshold` and `hls_threshold`. There're also `abs_threshold` etc for Sobel gradient but they're not used.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The `perspective_transform` function does the transform. The tricky part of this step is to find good `src` and `dst` coordinates so that the transformed image is accurate. The approach I take to find them is first pick 4 points in the untransformed image. I use the `straight_lines1.jpg` because I know what transformed image should look like. The function `retrieve_points_for_warping` is written to help print out coordinates when mouse click on a point. Then choose 4 points in destination image so that they form a rectangle. And then do the transform. If it shows awry in one end, then go back and change src coordinates accordinglly until it can be transformed into a pair of paralle lines.

This resulted in the following source and destination points:

```python
src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
dst = np.float32([[(320, 720), (320, 0), (920, 0), (920, 720)]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use the sliding window approach taught in the lecture to detect lane-line pixels. The `sliding_window_find` does that. Later, if a fit has been found, the `quick_find` will instead be used. 

However, using the fit directly leads to sudden change in curvatures. Also, in challenge videos, not all frame can be found having good lines due to limition of color thresholding. So that means we need store past fits and average them for these two purposes.

The `Tracker` class is written for this and to help debug problems. The `checkSanity` function will compare current left lane right lane (`left_fitx` and `right_fitx`) gap with historical ones; if they're too far away, the current ones will be discarded and the average ones will be used. The `avgFactor` is 10, namely to use the average of the past 10 frames to fit the polynomial.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `curvature_and_offset`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a ![link to my video result](video1)
Here's a ![link to my challenge video result](video2)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline can't handle the harder challenge video due to its sharp turns and light conditions. Also the performance of this pipeline is not good, due to poor implementation and optimization.

In the future, I will mostly focus on improving the `checkSanity` function so that it can better predict the fit for current frame. Also I'd like to see if the thresholding method can be improved or not.
