from flask import Flask, render_template,redirect,request, send_file,url_for,jsonify
import cv2
import numpy as np
from io import BytesIO
import base64
from IPython.display import display
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from skimage import io, transform, metrics
import os

app = Flask(__name__)

uploaded_files = {}

# Set the path to the Tesseract executable  # have to change this when rendering the application
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def apply_filter(input_image_path, filter_name):
    """
    Apply the specified filter to the input image and display the result.

    Args:
        input_image_path (str): Path to the input image.
        filter_name (str): Name of the filter to apply.
    """
    # Open the image file
    with Image.open(input_image_path) as img:
        # Apply the specified filter
        if filter_name == '2':
            filtered_image = img.filter(ImageFilter.CONTOUR)
        elif filter_name == '3':
            filtered_image = img.filter(ImageFilter.EDGE_ENHANCE)
        elif filter_name == '5':
            filtered_image = img.filter(ImageFilter.EMBOSS)
        elif filter_name == '6':
            filtered_image = img.filter(ImageFilter.SHARPEN)
        elif filter_name == '7':
            filtered_image = img.filter(ImageFilter.SMOOTH_MORE)
        elif filter_name == '4':
            filtered_image = img.convert('L')
        elif filter_name == '1':   # for contrast
            with Image.open(input_image_path) as img:
            # Apply histogram equalization to increase contrast
                filtered_image = ImageOps.equalize(img)

        else:
            raise ValueError("Invalid filter name. Choose from: BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EMBOSS, SHARPEN")

def rotate_image(img_path,degree):

    image1=Image.open(img_path)
    rotated_image=image1.rotate(degree)
    return rotated_image

# def extract_text(image_path):
#     # Path to the image have to be changed 
#     #image_path = "text_img.png"
#     img = Image.open(image_path)
#     extracted_text = pytesseract.image_to_string(img)
#     return extract_text

# def calculate_ssim(image1_path, image2_path):
#     # Load images
#     image1 = io.imread(image1_path, as_gray=True)
#     image2 = io.imread(image2_path, as_gray=True)
    
#     # Resize images to the same dimensions
#     min_height = min(image1.shape[0], image2.shape[0])
#     min_width = min(image1.shape[1], image2.shape[1])
#     image1 = transform.resize(image1, (min_height, min_width))
#     image2 = transform.resize(image2, (min_height, min_width))
    
#     # Calculate SSIM
#     ssim = metrics.structural_similarity(image1, image2, data_range=image2.max() - image2.min())
#     return str(int(ssim*100))+"%" 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            uploaded_files[filename] = file.read()
            return redirect(url_for('dashboard', filename=filename))
    return render_template('index.html')

@app.route('/dashboard/<filename>', methods=['GET', 'POST'])
def dashboard(filename):
    if filename in uploaded_files:
        if request.method == 'POST':
            if 'blur_radius' in request.form:
                blur_radius = int(request.form['blur_radius'])
                blur_image_in_memory(filename, blur_radius)
            elif 'left' in request.form:

                # Get the image data
                image_data = uploaded_files[filename]
                # Open the image using PIL
                image = Image.open(BytesIO(image_data))
                # Get the width of the image
                image_width = image.width
                image_height=image.height

                left = int(request.form['left'])
                top = int(request.form['top'])
                width1 = int(request.form['width'])
                width=(image_width/100)*width1
                height1 = int(request.form['height'])
                height=(image_height/100)*height1
                crop_image_in_memory(filename, left, top, width, height)
            elif 'rotate' in request.form:
                rotate_angle = int(request.form['rotate'])
                rotate_image_in_memory(filename, rotate_angle)  # Call rotate function
            
            elif 'filter_option' in request.form:
                filter_option = request.form['filter_option']
                apply_filter_in_memory(filename, filter_option)  # Call apply filter function

            
            return redirect(url_for('dashboard', filename=filename))
        
        encoded_image = base64.b64encode(uploaded_files[filename]).decode('utf-8')
        return render_template('dashboard.html', filename=filename, encoded_image=encoded_image)
    else:
        return "File not found"

def blur_image_in_memory(filename, blur_radius):
    image_data = uploaded_files[filename]
    image = Image.open(BytesIO(image_data))
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    buffered = BytesIO()
    blurred_image.save(buffered, format="PNG")
    uploaded_files[filename] = buffered.getvalue()

def crop_image_in_memory(filename, left, top, width, height):
    image_data = uploaded_files[filename]
    image = Image.open(BytesIO(image_data))
    cropped_image = image.crop((left, top, left + width, top + height))
    buffered = BytesIO()
    cropped_image.save(buffered, format="PNG")
    uploaded_files[filename] = buffered.getvalue()

def rotate_image_in_memory(filename, degree):
    image_data = uploaded_files[filename]
    image = Image.open(BytesIO(image_data))
    rotated_image = image.rotate(degree, expand=True)
    buffered = BytesIO()
    rotated_image.save(buffered, format="PNG")
    uploaded_files[filename] = buffered.getvalue()

def apply_filter_in_memory(filename, filter_option):
    image_data = uploaded_files[filename]
    image = Image.open(BytesIO(image_data))
    
    # Apply the selected filter
    if filter_option == "Contrast":
        filtered_image = image.filter(ImageFilter.CONTRAST)
    elif filter_option == "Contour":
        filtered_image = image.filter(ImageFilter.CONTOUR)
    elif filter_option == "Edge Enhance":
        filtered_image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif filter_option == "Grey":
        filtered_image = image.convert("L")  # Convert to grayscale
    elif filter_option == "Emboss":
        filtered_image = image.filter(ImageFilter.EMBOSS)
    elif filter_option == "Sharpen":
        filtered_image = image.filter(ImageFilter.SHARPEN)
    elif filter_option == "Smooth":
        filtered_image = image.filter(ImageFilter.SMOOTH)
    elif filter_option == "Find Edges":
        filtered_image = image.filter(ImageFilter.FIND_EDGES)  # Added this line
    else:
        # Default to original image if no valid filter option is selected
        filtered_image = image

    # Save the filtered image back to memory
    buffered = BytesIO()
    filtered_image.save(buffered, format="PNG")
    uploaded_files[filename] = buffered.getvalue()


 
# @app.route('/extract', methods=['POST'])
# def extract_text():
#     img=img_list[-1]

#     # Open the uploaded image
#     image = Image.open(img)

#     # Perform OCR to extract text from the image
#     extracted_text = pytesseract.image_to_string(image)

#     # Convert original image to base64 format
#     img1=img_list[-2]
#     img_io=BytesIO()
#     img1.save(img_io,format='JPEG')
#     img_io.seek(0)
#     img1 =base64.b64encode(img_io.getvalue()).decode('utf-8')

#     return render_template('dashboard.html', extract_text=extract_text,img1=img1)


if __name__ == '__main__':
    app.run(debug=True)
