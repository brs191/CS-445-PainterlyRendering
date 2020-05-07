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

class Painter:
    def __init__(self):
        print("Initialize Painter class")
        self.sample_img_dir = 'images/source1.jpg'
        self.sample_img = None

        # constant factors for various parameters in the paper.
        self.gaussian_factor = 1 #.5
        self.grid_factor = 1 #.5
        self.threshold = 10
        self.minStrokeLength = 2
        self.maxStrokeLength = 6
    
    def paint(self, sourceImage, brush_sizes): #, srcImg, R):
#        function paint(sourceImage,R1 ... Rn) {
#            canvas := a new constant color image
#            
#            // paint the canvasfor each brush radius Ri, from largest to smallest 
#            do {
#               // apply Gaussian blur
#               referenceImage = sourceImage * G(fσRi)
#               // paint a layer
#               paintLayer(canvas, referenceImage, Ri)
#            }
#            return canvas
#        }
        H, W, C = sourceImage.shape
        canvas = np.zeros((H,W,C)) 
        
        for r in brush_sizes:

            """
            Gaussian kernel size. ksize.width and ksize.height can differ 
            but they both must be positive and odd. Or, they can be zero's 
            and then they are computed from sigma. 
            """
            r_temp = self.gaussian_factor * r
            referenceImage = cv2.GaussianBlur(sourceImage, (r_temp, r_temp), cv2.BORDER_REFLECT)
            plt.figure()
            plt.imshow(referenceImage)
            plt.title(str(r) + " gaussian filtered")
            self.paintLayer(sourceImage, referenceImage, r)
        
        return canvas
    
        
    def paintLayer(self, canvas, referenceImage, r): #, canvas, refImg, R)
        print("Implement paintLayer function")
#        procedure paintLayer(canvas,referenceImage, R)
#        {
#            S := a new set of strokes, initially empty
#            
#            // create a pointwise difference image
#            D := difference(canvas,referenceImage)
#            
#            grid := f g R
#            
#            for x=0 to imageWidth stepsize grid do
#                for y=0 to imageHeight stepsize grid do
#                {
#                    // sum the error near (x,y)
#                    M := the region (x-grid/2..x+grid/2, y-grid/2..y+grid/2)
#                    
#                    areaError := ∑ i , j ∈ M D i,j / grid 2
#                    
#                    if (areaError > T) then
#                    {
#                    // find the largest error point
#                        (x 1 ,y 1 ) := arg max i , j ∈ M D i,j
#                        s :=makeStroke(R,x 1 ,y 1 ,referenceImage)
#                        add s to S
#                    }
#                }
#            paint all strokes in S on the canvas, in random order
#        }

        S = [] # Set of strokes, intitally empty
        """
        Image difference formula = 
        |(r1,g1,b1) – (r2,g2,b2)| = ((r1 – r2)2 + (g1 –g2)2 + (b1 – b2)2)1/2
        temp = (r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2
        sqrt(sum((temp, axis=-1))
        """        
        
#        D = np.sum((canvas[:,:,:3] - referenceImage[:,:,:3])**2, axis=2) 
        D = np.sqrt(np.sum((canvas[:,:,:3] - referenceImage[:,:,:3])**2, axis=2))
        grid = self.grid_factor * r
        print("grid size is ", grid)
        
        width, height, c = canvas.shape
        print("canvas shape ", canvas.shape)
        for x in range(0, width, grid): # start from (grid/2, width - grid/2, grid)
            for y in range(0, height, grid): # start from (grid/2, height - grid/2, grid)
                # but x starts from 0. need to fix below??
                # M = D[x-grid/2:x+grid/2, y-grid/2:y+grid/2]
                 M = D[x:x+grid, y:y+grid]
                 print("M shape is ", M.shape) # 15x15
                 areaError = np.sum(M)/(grid**2)
                 print("areaError threshold execced ", areaError)
                 if (areaError > self.threshold):
                     print("areaError threshold execced ", areaError)
                     # find the largest error point
                     x1, y1 = np.where(M == np.amax(M))
                     s = self.make_stroke(canvas, r, x + x1[0], y + y1[0], referenceImage) # For Part 1 - Normal strokes
                     #s = self.makeSplineStroke(canvas, r, x1[0], y1[0], referenceImage)
                     S.append(s)
        
        # paint all strokes in S on the canvas, in random order
        random_strokes = random.sample(S, len(S)) 
        print("Painting strokes")
        for stroke in random_strokes:
            canvas = cv2.circle(canvas,(stroke["y"],stroke["x"]), stroke["r"], (stroke["c1"],stroke["c2"],stroke["c3"]), -1)
        canvas = canvas.astype('uint8')
        plt.figure()
        plt.imshow(canvas)
        plt.title(str(r) + " printLayer")
        
        return canvas
        
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
        print("Implement makeSplineStroke function")
#        function makeSplineStroke(x 0 ,y 0 ,R,refImage)
#        {
#            strokeColor = refImage.color(x 0 ,y 0 )
#            K = a new stroke with radius R and color strokeColor
#            add point (x 0 ,y 0 ) to K
#            (x,y) := (x 0 ,y 0 )
#            (lastDx,lastDy) := (0,0)
#        
#            for i=1 to maxStrokeLength do
#            {
#                if (i > minStrokeLength and
#                    |refImage.color(x,y)-canvas.color(x,y)|<
#                    |refImage.color(x,y)-strokeColor|) then
#                    return K
#                
#                // detect vanishing gradient
#                if (refImage.gradientMag(x,y) == 0) then
#                    return K
#                
#                // get unit vector of gradient
#                (gx,gy) := refImage.gradientDirection(x,y)
#                // compute a normal direction
#                (dx,dy) := (-gy, gx)
#                
#                // if necessary, reverse direction
#                if (lastDx * dx + lastDy * dy < 0) then
#                    (dx,dy) := (-dx, -dy)
#                    
#                // filter the stroke direction
#                (dx,dy) :=f c *(dx,dy)+(1-f c )*(lastDx,lastDy)
#                (dx,dy) := (dx,dy)/(dx 2 + dy 2 ) 1/2
#                (x,y) := (x+R*dx, y+R*dy)
#                (lastDx,lastDy) := (dx,dy)
#                
#                add the point (x,y) to K
#            }
#            return K
#        }
        """
        The stroke isrepresented as a list of control points, a color, and 
        a brushradius. The control point (x0,y0) is added to the spline, 
        and thecolor of the reference image at (x0,y0) is used as the 
        # color of thespline.
        """
        strokeColor = refImage[x0, y0]
        print("strokeColor is ", strokeColor) # 137, 146, 167
        
        K = []
        K.append((x0,y0))
        print("K is ", K) # [(4,11)]
        print("K len is ", len(K)) # 1
        (x, y) = (x0, y0)
        (lastDx, lastDy) = (0, 0)
        
        for i in range(1, self.maxStrokeLength):
            a = abs(refImage[x,y] - canvas[x,y]) # this is a RGB value
            b = abs(refImage[x,y] - strokeColor) # this is a RGB value
            if (i > self.minStrokeLength and a > b): # don't think this works??
                return K
            
            #detect vanishing gradient
            
            
         

        return K

test = 2
def mytestcode():
    print("hello testcode ")
    arr = np.arange(9).reshape(3,3)
    ones = np.ones((3,3), dtype=int)
    print(arr)
    print(ones)
    print(arr.shape)
    print("sum is ", np.sum(ones)/3)
    
    print("max value is ", np.amax(arr))
    x1, y1 = np.where(arr == np.amax(arr))
    print("co-ord of max value is ", x1[0], ":", y1[0])
    
    (x, y) = (10, 20)
    print("x , y is ", x, y)

    sample_img = cv2.imread('images/source1.jpg')
    strokeColor1 = sample_img[4,11]
    print("strokeColor1 ", strokeColor1) #[168 145 137]
    val1 = np.sqrt(strokeColor1.dot(strokeColor1))
    print("val1 is ", val1)
    print("val11 is ", np.linalg.norm(strokeColor1))
    
    strokeColor2 = sample_img[11,4]
    print("strokeColor2 ", strokeColor2) #[192 145 114]
    val2 = np.sqrt((strokeColor2*strokeColor2).sum(axis=0)) 
    print("val2 is ", val2)
    print("val22 is ", np.linalg.norm(strokeColor2))
    print("storke difference is ", strokeColor1 - strokeColor2)
    
    
    
def main():

    if test == 1:
        mytestcode()
    else:
        painter = Painter()
       
        print("sample img dir", painter.sample_img_dir)
        if os.path.exists(painter.sample_img_dir):
            sample_img = cv2.imread(painter.sample_img_dir)
            img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
            print("sample_img shape ", sample_img.shape)
            plt.imshow(img_rgb)
            plt.title('sample image')
    
        brush_sizes = [5]
    #    brush_sizes = [10]
        canvas = painter.paint(img_rgb, brush_sizes)
    #    painter.paintLayer(canvas, img_rgb, brush_sizes)
        painter.makeSplineStroke()
        print("Program ended")
    
if __name__ == "__main__":
    main()
