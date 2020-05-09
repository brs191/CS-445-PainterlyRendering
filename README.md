# CS-445-PainterlyRendering
### CS 445 Final project on Painterly Rendering 
 - Based on the research paper "Painterly Rendering with Curved Brush Strokes of Multiple Sizes" by
    Aaron Hertzmann
    Media Research Laboratory
    Department of Computer Science
    New York University
    https://www.mrl.nyu.edu/publications/painterly98/hertzmann-siggraph98.pdf


### Setup
Below dependencies need to installed:
```
opencv-python
numpy
matplotlib
```
### Project structure
- painterly_rendering.py is the main file which contains the code.
- images/input folder contains the images that are used to test the code.
- images/output folder contains the images that are rendered using normal brush strokes algorithm and curved brush strokes algorithm.

### Implementation Details:
- This project is developed using python. Python libraries used are:
  opencv-python
  numpy
  matplotlib
- Predefined functions used:
  cv2.GaussianBlur() - to blur an image using a Gaussian filter.
  cv2.Sobel() - to get the unit vector of gradient.
  cv2.line(), cv2.circle() - to produce strokes on the image.

- User Defined class and functions used:
  We created a class Painter that implements the apis Paint() as PaintLayer() as described in the paper. With respect to Stroke selections we have defined two methods, MakeStroke() to draw normal strokes and MakeSplineStroke() to draw curved strokes as described in the paper. 

### Running the file
```
python painterly_rendering.py
```

