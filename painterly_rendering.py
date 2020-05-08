#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:45:40 2020

@author: hanisha nunna, raja shekar bollam
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

NORMAL_STROKE = 1 
SPLINE_STROKE = 2

class Painter:
    def __init__(self):
        self.input_dir = 'images/input/'
        self.output_dir = 'images/output/'
        self.src_file = 'empty'
        self.sample_img = None

        # constant factors for various parameters in the paper.
        self.gaussian_factor = .5
        self.grid_factor = 1 #.5
        self.strokeFilterConstant = 1
        self.threshold = 10
        self.minStrokeLength = 2
        self.maxStrokeLength = 6
    
    def paint(self, sourceImage, brush_sizes, strokeType):
        brush_sizes = sorted(brush_sizes, reverse=True)
      
        H, W, C = sourceImage.shape
        canvas = np.zeros((H,W,C)) 
        
        for r in brush_sizes:
            """
            Gaussian kernel size. ksize.width and ksize.height can differ 
            but they both must be positive and odd. Or, they can be zero's 
            and then they are computed from sigma. 
            """
            r_temp = int(self.gaussian_factor * r)
            if (r_temp %2 == 0):
                r_temp = r_temp + 1
            referenceImage = cv2.GaussianBlur(sourceImage, (r_temp, r_temp), cv2.BORDER_REFLECT)
            
            gauss_img = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.output_dir + "gaussian_brush_" + str(r) + "_" + self.src_file, gauss_img)
            
            self.paintLayer(canvas, referenceImage, r, strokeType)
        
    def paintLayer(self, canvas, referenceImage, r, strokeType):
        S = [] # Set of strokes, intitally empty
        """
        Image difference formula = 
        |(r1,g1,b1) – (r2,g2,b2)| = ((r1 – r2)2 + (g1 –g2)2 + (b1 – b2)2)1/2
        temp = (r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2
        sqrt(sum((temp, axis=-1))
        """        
        D = np.sqrt(np.sum((canvas[:,:,:3] - referenceImage[:,:,:3])**2, axis=2))
        grid = self.grid_factor * r
        
        width, height, c = canvas.shape
        for x in range(0, width, grid):
            for y in range(0, height, grid):
                 M = D[x:x+grid, y:y+grid]
                 areaError = np.sum(M)/(grid**2)
                 if (areaError > self.threshold):
#                     print("areaError threshold execced ", areaError)
                     # find the largest error point
                     x1, y1 = np.where(M == np.amax(M))
                     if strokeType == SPLINE_STROKE:
                         # For Part 2 - Spline Strokes
                         s = self.makeSplineStroke(canvas, r, x + x1[0], y + y1[0], referenceImage)
                     else:
                         # For Part 1 - Normal strokes
                         s = self.make_stroke(canvas, r, x + x1[0], y + y1[0], referenceImage)
                     S.append(s)
        
        # paint all strokes in S on the canvas, in random order
        random_strokes = random.sample(S, len(S)) 
        if strokeType == SPLINE_STROKE:
            for stroke in random_strokes:
                stroke1_points = stroke[0]["points"]
                p1 = stroke1_points[0]
                if (len(stroke1_points) > 1):
                    p2 = stroke1_points[1]
                else:
                    p2 = p1
                colour_array = stroke[0]["color"]
                colour = (int(colour_array[0]),int(colour_array[1]),int(colour_array[2]))
                canvas = cv2.line(canvas,p1,p2,colour,stroke[0]["r"])
                ss_img = cv2.cvtColor(canvas.astype(np.float32), cv2.COLOR_BGR2RGB)
                cv2.imwrite(self.output_dir + "splineStroke_brush_" + str(r) + "_" + self.src_file, ss_img)
        else:
            for stroke in random_strokes:
                canvas = cv2.circle(canvas,(stroke["y"],stroke["x"]), stroke["r"], (stroke["c1"],stroke["c2"],stroke["c3"]), -1)
            ns_img = cv2.cvtColor(canvas.astype(np.float32), cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.output_dir + "normalStroke_brush_" + str(r) + "_" + self.src_file, ns_img)
            
        canvas = canvas.astype('uint8')
        plt.figure()
        plt.imshow(canvas)
        plt.title(str(r) + " printLayer")
        
    def make_stroke(self, canvas, r, x, y, reference_img):
        color = reference_img[x, y]
        stroke_value = {
            "x": x,
            "y": y,
            "r": r,
            "c1":int(color[0]),
            "c2":int(color[1]),
            "c3":int(color[2])
        }
        return stroke_value

    def makeSplineStroke(self, canvas, r, x0, y0, refImage):
        """
        The stroke is represented as a list of control points, a color, and 
        a brushradius. The control point (x0,y0) is added to the spline, 
        and thecolor of the reference image at (x0,y0) is used as the 
        # color of thespline.
        """
        strokeColor = refImage[x0, y0]
       
        K = []
        control_points = []
        control_points.append((y0,x0))
        spline_stroke_value = {
            "points": control_points,
            "r": r,
            "color":strokeColor
        }
        K.append(spline_stroke_value)
        (x, y) = (x0, y0)
        (lastDx, lastDy) = (0, 0)
        
        for i in range(1, self.maxStrokeLength):
            a = abs(refImage[x,y] - canvas[x,y])
            b = abs(refImage[x,y] - strokeColor)
            
            if (i > self.minStrokeLength and a.all() > b.all()):
                return K
            
            #detect vanishing gradient
            if (refImage[x,y].any() == 0):
                return K
            
            grayImage = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
        
            # get unit vector of gradient
            grad_x = cv2.Sobel(grayImage, cv2.CV_32F, 1, 0)
            grad_y = cv2.Sobel(grayImage, cv2.CV_32F, 0, 1)
            
            gx = grad_x[x, y]
            gy = grad_y[x, y]
            
            # compute a normal direction
            dx,dy = -gy, gx

            # if necessary, reverse direction
            if ((lastDx * dx).any() + (lastDy * dy).any() < 0):
                dx,dy = -dx, -dy
                
            # filter the stroke direction
            dx = self.strokeFilterConstant * dx + (1-self.strokeFilterConstant) * lastDx
            dy = self.strokeFilterConstant * dy + (1-self.strokeFilterConstant) * lastDy
                
            dx = dx / np.sqrt(dx**2+dy**2)
            dy = dy / np.sqrt(dx**2+dy**2)

            final_x, final_y = x + r*dx, y + r*dy
            lastDx, lastDy = dx, dy
            
            if(not np.isnan(final_x) and not np.isnan(final_y)):
                spline_stroke_value["points"].append((int(round(final_y)), int(round(final_x))))
        return K
    
def main():

    painter = Painter()

    brush_sizes = [5, 13]
    src_file = 'source1.jpg'
    
    painter.src_file = src_file
    print(".. ", painter.input_dir + src_file)
    if os.path.exists(painter.input_dir + src_file):
        sample_img = cv2.imread(painter.input_dir + src_file)
        img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(painter.output_dir + "original_" + src_file, sample_img)

        painter.paint(img_rgb, brush_sizes, NORMAL_STROKE)
        painter.paint(img_rgb, brush_sizes, SPLINE_STROKE)
        

    brush_sizes = [3, 5]
    src_file = 'applesNoranges.jpg'
    painter.src_file = src_file
    print(".. ", painter.input_dir + src_file)
    if os.path.exists(painter.input_dir + src_file):
        sample_img = cv2.imread(painter.input_dir + src_file)
        img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(painter.output_dir + "original_" + src_file, sample_img)

        painter.paint(img_rgb, brush_sizes, NORMAL_STROKE)
        painter.paint(img_rgb, brush_sizes, SPLINE_STROKE)

    brush_sizes = [5, 7]
    src_file = 'scenary.jpg'
    painter.src_file = src_file
    print(".. ", painter.input_dir + src_file)
    if os.path.exists(painter.input_dir + src_file):
        sample_img = cv2.imread(painter.input_dir + src_file)
        img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(painter.output_dir + "original_" + src_file, sample_img)

        painter.paint(img_rgb, brush_sizes, NORMAL_STROKE)
        painter.paint(img_rgb, brush_sizes, SPLINE_STROKE)
            
if __name__ == "__main__":
    main()
