import numpy as np
import matplotlib.pyplot as plt
import cv2
# from skimage.color import bgr2gray
import os

from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image

os.chdir('./code/')

# Defining global kernels which will be used in the corner detection algorithm
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

gauss_blur = np.array([[1/16, 2/16, 1/16],
                       [2/16, 4/16, 2/16],
                       [1/16, 2/16, 1/16]])


def bgr2gray(image):
    """
    Converts an image in BGR format to grayscale.

    Args:
        image (np.ndarray): has to be in BGR format

    Returns:
        np.ndarray: grayscale image
    """
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return gray


def convolve(image: np.ndarray, kernel: np.ndarray, padding: bool = False):
    """
    Performs window convolution of the argument `image` using the provided `kernel`. If padding is
    set to `True`, the output image shape will be the same as that of the input `image`.

    Args:
        image (np.ndarray): input image
        kernel (np.ndarray): kernel
        padding (bool, optional): if set to true, pads the image with zeros in all dimensions such that
        output after convolution has the same shape as the input image. Defaults to False.

    Returns:
        np.ndarray: convolved image after sliding the input `kernel` over the input `image`. 
    """

    k = kernel.shape[0]

    if padding:
        out_size = image.shape[0]
    else:
        out_size = image.shape[0] - k + 1

    if padding:
        image = np.pad(image, k//2)

    convolved = np.empty((out_size, out_size), dtype=np.float64)
    for x in range(out_size):
        for y in range(out_size):
            sub_img = image[x:x+k, y:y+k]
            convolved[x, y] = np.sum(np.multiply(sub_img, kernel))

    return convolved


def harris(image: np.ndarray, sensitivity: np.float64 = 0.06, threshold: np.float64 = 0.4):
    """
    Performs corner detection using the harris corner detection algorithm.

    Args:
        image (np.ndarray): image for which corners need to be detected
        sensitivity (np.float64, optional): harris sensitivity parameter, should be close to 
        zero. Defaults to 0.06.
        threshold (np.float64, optional): harris threshold parameter. Defaults to 0.4.

    Returns:
        np.ndarray: array containing coordinates of all points which have corners, as detected
        by the algorithm 
    """
    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)
    grad_x2 = np.square(grad_x)
    grad_y2 = np.square(grad_y)
    grad_xy = grad_x*grad_y

    gaus_x2 = convolve(grad_x2, gauss_blur)
    gaus_y2 = convolve(grad_y2, gauss_blur)
    gaus_xy = convolve(grad_xy, gauss_blur)

    harris_fn = gaus_x2*gaus_y2 - \
        np.square(gaus_xy) - sensitivity*np.square(gaus_x2+gaus_y2)
    cv2.normalize(harris_fn, harris_fn, 0, 1, cv2.NORM_MINMAX)

    loc = np.where(harris_fn > threshold)
    pts = np.array([pt for pt in zip(*loc[::-1])])

    return pts


def load_image_to_canvas(path: str):
    """
    Function to load an image into the tkinter window.

    Args:
        path (str): path to image
    """
    img = ImageTk.PhotoImage(Image.open(path))
    (width, height) = plt.imread(path).shape[:2]
    canvas = Canvas(root, width=width+20, height=height+20)
    canvas.pack()
    canvas.create_image(20, 20, anchor=NW, image=img)
    for button in buttons:
        button.configure(state="disabled")

    B = ttk.Button(root, text="Detect Corners",
                   command=lambda path=path: get_inputs(path))
    B.pack()
    ttk.Button(root, text="Clear", command=clear_frame).pack(pady=5)
    root.mainloop()


def get_inputs(path: str):
    """
    Takes sensitivity and threshold as inputs from the tkinter window and then runs 
    the harris corner detection algorithm

    Args:
        path (str): path to image
    """
    global new_tk
    new_tk = Toplevel()
    new_tk.title("Selecting parameters")

    # Defining function to accept inputs from entry fields
    def submit():
        sensitivity = s_entry.get()
        threshold = t_entry.get()
        new_tk.title("Harris corner detection output")
        for widgets in new_tk.winfo_children():
            widgets.destroy()
        run_harris(path, float(sensitivity), float(threshold))

    # Create label and entry widget for taking sensitivity as input
    s_label = Label(new_tk, text="Sensitivity (default = 0.06)")
    s_label.pack()
    s_entry = Entry(new_tk, width=40)
    s_entry.focus_set()
    s_entry.pack()

    # Create label and entry widget for taking threshold as input
    t_label = Label(new_tk, text="Threshold (default = 0.4)")
    t_label.pack()
    t_entry = Entry(new_tk, width=40)
    t_entry.pack()

    # Create a button to accept inputs from entry fields
    ttk.Button(new_tk, text="Submit", width=20, command=submit).pack(pady=20)
    new_tk.mainloop()


def run_harris(path: str, sensitivity: np.float64 = 0.06, threshold: np.float64 = 0.4, save_img: bool = True):
    """
    Function to run the harris corner detection function and display the output 
    in a new tkinter window.

    Args:
        path (str): path to image
        save_img (bool, optional): saves the image with marked corners if set to true. Defaults to True.
    """
    image_bgr = plt.imread(path)
    image = bgr2gray(image_bgr)
    detected_corners = harris(
        image, sensitivity=sensitivity, threshold=threshold)

    # Plotting corner points and saving the image to a temporary file "fig.jpg"
    fig = plt.figure()
    plt.imshow(image_bgr, cmap='gray')
    plt.plot(detected_corners[:, 0],
             detected_corners[:, 1], '.r', markersize=5)
    plt.axis('off')
    plt.savefig('fig.jpg', bbox_inches='tight', dpi=96)

    # Sanity check for sensitivity and threshold used
    Label(new_tk, text="Sensitivity used = " + str(sensitivity)).pack()
    Label(new_tk, text="Threshold used = " + str(threshold)).pack()

    image = ImageTk.PhotoImage(Image.open('fig.jpg'))           # Load image
    # get image dimensions
    (width, height) = plt.imread('fig.jpg').shape[:2]
    # Create canvas using the image dimensions
    canvas = Canvas(new_tk, width=width+20, height=height+20)
    canvas.pack()
    # Add image to the canvas
    canvas.create_image(20, 20, anchor=NW, image=image)

    # Delete the temporary image file created
    # os.system('del fig.jpg')
    # Run the new window with the modifications
    new_tk.mainloop()


def clear_frame():
   for widgets in root.winfo_children():
      widgets.destroy()


def main():
    global root, buttons
    list_files = os.listdir('./images/')
    # Defining the tkinter window instance
    root = Tk()
    root.title("Image selection")
    Label(root, text="Greetings! Please choose an image.").pack(side=TOP, pady=10)

    # Creating buttons for all files present in the images/ directory
    buttons = []
    for file in list_files:
        path = './images/' + file
        B = ttk.Button(root, text=file,
                       command=lambda path=path: load_image_to_canvas(path))
        B.pack()
        buttons.append(B)

    # Running the tkinter instance
    root.mainloop()


if __name__ == "__main__":
    main()
