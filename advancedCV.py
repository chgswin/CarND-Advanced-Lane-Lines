import cv2
import numpy as np
import matplotlib.pyplot as plt

def undistort_image(image, matrix, distance):
    return cv2.undistort(image, matrix, distance, None, matrix)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh = (0, 255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    
    try:
        #grayscale the image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if orient=='y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize= sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize= sobel_kernel)
        
        #Take the absolute value of the derivate or gradient
        abs_sobel = np.absolute(sobel)
        
        scaled_sobel = np.uint8(255*abs_sobel / np.max(abs_sobel))
        
        binary_output = np.zeros_like(scaled_sobel)
        
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        
    except Exception as e:
        print(e)
        
        binary_output = None
    
    finally:
        return binary_output

def mag_thresh(img, sobel_kernel = 3, thresh = (0, 255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        sobel = (sobel_x**2 + sobel_y**2)**0.5
        
        scaled_sobel = np.uint8(sobel*255/np.max(sobel))
        
        binary_output = np.zeros_like(scaled_sobel)
        
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)]=1
            
    except Exception as e:
        
        print(e)
        binary_output = None
    
    finally:
        return binary_output
    
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    thresh_min, thresh_max = thresh[0], thresh[1]
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #Calculate gradient direction
        #Apply threshold
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        sobel_angle = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))

        binary_output = np.zeros_like(sobel_angle)

        #binary mask where direction thresholds are met
        binary_output[(sobel_angle >= thresh_min) & (sobel_angle <= thresh_max)] = 1
    except Exception as e:
        print(e)
        binary_output = None
        
    finally:
        return binary_output
    
def transform_source(width, height, width_offset, ref_width = 512, ref_height = 460):
    
    return np.float32([[width_offset, height],[ref_width, ref_height],[width - ref_width, ref_height],[width - width_offset, height]])
    
def transform_destination(width, height, width_offset):
    
    return np.float32([[width_offset, height], [width_offset, 0], [width - width_offset, 0], [width - width_offset, height]])

#transform the image
def warp(img, src, dst):
    
    width = img.shape[1]
    height = img.shape[0]
    
    #get the perspective transform matrix
    perspective_transform = cv2.getPerspectiveTransform(src, dst)

    #get the inverse perspective transform matrix
    #for the original image
    inverse_perspective_transform = cv2.getPerspectiveTransform(dst, src)
    
    #get the transformed (or warped) result
    warped = cv2.warpPerspective(img, perspective_transform, (width, height), flags=cv2.INTER_LINEAR)    
    
    return warped, perspective_transform, inverse_perspective_transform

    
def threshold_image(img):
    #Color channel only
    combined_color = np.zeros((img.shape[0], img.shape[1]))

    RGB_red = img[:,:,0]
    road_dst_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    HLS_sat = road_dst_HLS[:,:,2]
    road_dst_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    HSV_val = road_dst_HSV[:,:,2]

    red_thres = (175, 255)
    val_thres = (170, 255)
    sat_thres = (100, 255)

    color_criteria = ((RGB_red >= red_thres[0]) & (RGB_red <= red_thres[1])) \
                    & ((HLS_sat >= sat_thres[0]) & (HLS_sat <= sat_thres[1])) #
                    # ((HSV_val >= val_thres[0]) & (HSV_val <= val_thres[1])) 


    combined_color[color_criteria] = 1
    
    #Gradient color only
    combined_gradient = np.zeros((img.shape[0], img.shape[1]))

    sobelx_channel = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh = (40, 255))
    sobely_channel = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh = (60, 255))

    mag_channel = mag_thresh(img, 3, (60, 255))


    dir_channel = dir_thresh(img, 3, (40 * np.pi / 180,75 * np.pi / 180))

    gradient_criteria = ((sobelx_channel == 1) & (sobely_channel == 1) & (mag_channel == 1)) \
                        & (dir_channel == 1)

    combined_gradient[gradient_criteria] = 1
    
    #combine two channels
    combined = np.zeros_like(combined_gradient) 
    combined[(combined_gradient == 1) | (combined_color == 1)] = 255
    
    return combined

def histogram(img):
    #Grab the bottom half of the image
    
    bottom_half = img[img.shape[0] // 2 :, :] / 255
    
    #axis = 0: vertical
    #axis = 1: horizontal
    result = np.sum(bottom_half, axis = 0)

    return result

def scan_lane_pixels(warped_img, nwindows = 10, margin = 50, minpix = 100):
    #create an output image to draw on and visualize the result
    out_img = np.uint8(np.dstack((warped_img, warped_img, warped_img)))
    
    #find the peak of the left and right halves of the histogram.
    #they will be the starting points for the left and right lines.
    hist = histogram(warped_img)
    
    midpoint = np.int(hist.shape[0]//2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint

    #Height of windows = height of image / number of windows
    window_height = np.int(warped_img.shape[0] // nwindows)

    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #current position to be updated later for each winddow in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    #Lists for left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    left_window = []
    right_window = []
    
    for window in range(nwindows):
        #Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window+1) * window_height
        win_y_high = warped_img.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        left_window.append(((win_xleft_low, win_y_low), (win_xleft_high, win_y_high)))
        right_window.append(((win_xright_low, win_y_low),(win_xright_high, win_y_high)))

        #Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        #Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        #if more than minpix pixels are found, next window will be on their mean position
        if len(good_left_inds) >= minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) >= minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    #Concate the array of indices (list of list of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)        
        right_lane_inds = np.concatenate(right_lane_inds)
    
    except Exception as e:
        print(e)
        pass
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    out_img[lefty, leftx] = [0,255,0]
    out_img[righty, rightx] = [0,0,255]
    
    return leftx, lefty, rightx, righty, out_img, left_window, right_window

def search_around_poly(warped_img, last_left_fit_coeff, last_right_fit_coeff, nwindows = 9, margin = 100, minpix = 100):
    
    out_img = np.uint8(np.dstack((warped_img, warped_img, warped_img)))
    
    nonzero = warped_img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    
    left_midline = np.polyval(last_left_fit_coeff, nonzeroy)
    right_midline = np.polyval(last_right_fit_coeff, nonzeroy)
    
    
    left_lane_inds = (abs(left_midline - nonzerox) <= margin)
    right_lane_inds = (abs(right_midline - nonzerox) <= margin) 
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(leftx, lefty, rightx, righty, useWeight=True):
    
    try:
        if useWeight:
            weight = lefty**2
        else:
            weight = None
            
        left_fit_coeffs = np.polyfit(lefty, leftx, deg = 2, w=weight)
    except Exception as e:
        print("Left lane: ", e)
        left_fit_coeffs = None
        
    try:
        if useWeight:
            weight = righty**2
        else:
            weight = None
            
        right_fit_coeffs = np.polyfit(righty, rightx, deg = 2, w=weight)
        
    except Exception as e:
        print("Right lane: ", e)
        right_fit_coeffs = None
    
    return left_fit_coeffs, right_fit_coeffs

def draw_detected_lane_lines(warped_img, leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty, margin = 100):
    
    out_img = np.uint8(np.dstack((warped_img, warped_img, warped_img)))
    
    window_img = np.zeros_like(out_img)
    
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 255, 0]
    
    #Generate a polygon to illustrate the search window area
    #and recast the xand y points into usable format for cv2.fillPoly()
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
    mid_line = np.array([np.transpose(np.vstack([(left_fitx + right_fitx) / 2, ploty]))])
    cv2.polylines(window_img, np.int_([left_line]), isClosed = False, color = (255, 0, 0), thickness = 10)
    cv2.polylines(window_img, np.int_([right_line]), isClosed = False, color = (255, 0, 0), thickness = 10)
    cv2.polylines(window_img, np.int_([mid_line]), isClosed = False, color = (255,0,0), thickness = 15)
    
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])

    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    #Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0,255))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,0,255))

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result

def fill_lane(warped_img, leftx, lefty, rightx, righty, 
              left_fitx, right_fitx, ploty, 
              left_window, right_window, searchAroundPoly = True):
    
    out_img = np.uint8(np.dstack((warped_img, warped_img, warped_img)))
    
    window_img = np.zeros_like(out_img, dtype=np.uint8)
    
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 255, 0]
    
    #Generate a polygon to illustrate the search window area
    #and recast the xand y points into usable format for cv2.fillPoly()
    
    left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
    mid_line = np.array([np.transpose(np.vstack([(left_fitx + right_fitx) / 2, ploty]))])
    
    #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack[right_fitx + margin, ploty]))])
    
    lane = np.hstack((left_line, right_line))
    
                  
    #Draw the lane onto the warped blank image

    cv2.fillPoly(window_img, np.int_([lane]), (0,255,0) if searchAroundPoly else (255,255,0))
    
    cv2.polylines(window_img, np.int_([left_line]), isClosed = False, color = (255, 0, 0), thickness = 10)
    cv2.polylines(window_img, np.int_([right_line]), isClosed = False, color = (255, 0, 0), thickness = 10)
    cv2.polylines(window_img, np.int_([mid_line]), isClosed = False, color = (255,0,0), thickness = 15)
    
    for low, high in left_window:
        cv2.rectangle(window_img, (low[0], low[1]), (high[0], high[1]), (255,0,0), 5)
    
    for low, high in right_window:
        cv2.rectangle(window_img, (low[0], low[1]), (high[0], high[1]), (255,0,0), 5)
    
    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return window_img

def measure_curvature_pixels_by_coeffs(ploty, left_fit_coeffs, right_fit_coeffs):
    
    y_eval = np.max(ploty)
    
    #Calculation of R_curve
    A_l, B_l, C_l = left_fit_coeffs[0], left_fit_coeffs[1], left_fit_coeffs[2]
    A_r, B_r, C_r = right_fit_coeffs[0], right_fit_coeffs[1], right_fit_coeffs[2]
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = (1+(2*A_l*y_eval + B_l) ** 2) ** (1.5) / abs(2*A_l)  ## Implement the calculation of the left line here
    right_curverad = (1 + (2*A_r*y_eval + B_r) ** 2) ** (1.5) / abs(2*A_r)  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

def measure_curvature_pixels_by_data(ploty, left_fitx, right_fitx):
    left_fit_coeffs = np.polyfit(ploty, left_fitx, 2)
    right_fit_coeffs = np.polyfit(ploty, right_fitx, 2)

    return measure_curvature_pixels_by_coeffs(ploty, left_fit_coeffs, right_fit_coeffs)

def measure_curvature_real_by_data(ploty, left_fitx, right_fitx, xm_per_pix = 3.7/700, ym_per_pix = 30/720):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
   
    # Make sure to feed in your real data instead in your project!
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2) 
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    A_l, B_l, C_l = left_fit_cr[0], left_fit_cr[1], left_fit_cr[2]
    A_r, B_r, C_r = right_fit_cr[0], right_fit_cr[1], right_fit_cr[2]
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    y_real = y_eval * ym_per_pix
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = (1+(2*A_l*y_real + B_l) ** 2) ** (1.5) / abs(2*A_l)  ## Implement the calculation of the left line here
    right_curverad = (1 + (2*A_r*y_real + B_r) ** 2) ** (1.5) / abs(2*A_r)  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

#Assume that the camera is at the center of the car (horizontally)
def measure_distance_from_lane_center(imshape, left_fit_coeffs, right_fit_coeffs, xm_per_pix = 3.7/700):
    
    width = imshape[1]
    height = imshape[0]
    
    left_lane_pt = np.polyval(left_fit_coeffs, height)
    
    right_lane_pt = np.polyval(right_fit_coeffs, height)
    
    lane_center = np.int_((left_lane_pt + right_lane_pt) // 2)
    
    return (imshape[1]//2 - lane_center)*xm_per_pix


    