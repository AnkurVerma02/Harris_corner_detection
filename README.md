# Harris Corner Detection
***

Main libraries (with versions) used :
```
matplotlib==3.4.2
numpy==1.21.4
opencv_python==4.5.3.56
pillow==8.3.1
```
## Running the application
---
1. Directly double click on harris.exe file to run the application, no need to change the directory as directory has been initialised in the python script.
2. Choose any of the images for which you want to detect the corners and then enter suitable values for sensitivity and threshold.

## Running the python file, 
---
1. Navigate to the code directory in terminal/command prompt
2. Options: 
   * Run `python harris.py`.
   * Open the jupyter notebook `harris.ipynb` and run all cells. The GUI window will open automatically after the last cell is executed.

The directory structure of the repository is given below:

```
.
├── LICENSE
├── README.md
├── harris.exe
└── code
    ├── harris.ipynb
    ├── harris.py
    ├── requirements.txt
    ├── images
    │   ├── box.jpg
    │   ├── cat.jpeg
    │   ├── img_from_slides.png
    │   └── satellite_img.png
    └── output_images
        ├── satellite_corners1.jpg
        ├── satellite_corners2.jpg
        ├── satellite_corners3.jpg
        ├── slides_corners1.jpg
        ├── slides_corners2.jpg
        └── slides_corners3.jpg
```
## Sensitivity & threshold
---
The default values for sensitivity is 0.06 and that for threshold is 0.4.
As we increase the value of sensitivity, it detects more corners and on increasing the threshold it detects less number of corners this is because we are considering those values of the ‘R’ function (defined as ‘harris_fn’ in the code) which are above the threshold as the points to be considered for corners.

## Presentation
---
In the presentation the output images have been included along with the values of the threshold and sensitivity, we can see that the trend is observed as stated above on changing the values of threshold and sensitivity.

## Output images
---
In the output images the corners are marked using small red colored circles instead of pixels so it doesn't imply that the algorithm is detecting a region of pixels as the corners, it is classifying a particular pixel as a corner then that pixel is labelled using a small red circualr blob.