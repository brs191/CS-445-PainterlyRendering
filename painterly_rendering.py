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

class Painter:
    def __init__(self):
        print("Initialize Painter class")
        self.sample_img_dir = 'images/source1.jpg'
        self.sample_img = None
        
    
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
        H,W,C = sourceImage.shape
        canvas = np.zeros((H,W,3)) 
        
        # Sort the brush sizes from largest to smallest
        R = sorted(brush_sizes, reverse=True)
        
        #for r in R:
            #referenceImage = cv2.GaussianBlur(sourceImage,(r,r),2) # Constant value to be changed
            # paint a layer
            #canvas = paintLayer(canvas, referenceImage, r)
        
        # first brush size
        R1 = R[0]
        
        #Apply gaussian blur to source image
        blur = cv2.GaussianBlur(sourceImage,(R1,R1),2)
        plt.figure()
        plt.imshow(blur)
        plt.title('paint')
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
        D = np.sum((canvas[:,:,:3] - referenceImage[:,:,:3])**2,axis=2)
        
        print(S)
        print(D)
        plt.figure()
        plt.imshow(D)
        plt.title('paintLayer')
        
        return canvas
        
    def makeSplineStroke(self): #, x0, y0, refImage):
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

def main():

    painter = Painter()
   
    print("sample img dir", painter.sample_img_dir)
    if os.path.exists(painter.sample_img_dir):
        sample_img = cv2.imread(painter.sample_img_dir)
        img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title('sample image')

    brush_sizes = [10,5,15]            
    canvas = painter.paint(img_rgb, brush_sizes)
    painter.paintLayer(canvas, img_rgb, brush_sizes)
    painter.makeSplineStroke()
    print("Program ended")
    
if __name__ == "__main__":
    main()
