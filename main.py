import numpy as np
import pydicom
import sys
from PIL import Image
from io import BytesIO
import base64
import json
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from flask import Flask, jsonify
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class ImageExtractor(Resource):
    def get(self, file_name):
        return extract_images(file_name)

api.add_resource(ImageExtractor, '/<file_name>')


def extract_images(file_name, auto_invert=True):
    try:
        # Load the DICOM file
        dicom_data = pydicom.dcmread(getFileFullPath(file_name))

        # Apply modality LUT (if present)
        image = apply_modality_lut(dicom_data.pixel_array, dicom_data)

        # Apply VOI LUT (if present)
        image = apply_voi_lut(image, dicom_data)

        # Normalize the image array to 0-255
        image = (np.clip(image, 0, np.max(image)) / np.max(image) * 255).astype(np.uint8)

        # Automatically determine whether to invert the image
        if auto_invert:
            # Evaluate the number of pixels near the maximum and minimum
            low_pixel_count = np.sum(image < 50)  # Count of dark pixels
            high_pixel_count = np.sum(image > 205)  # Count of light pixels

            # Invert if there are more high-intensity pixels than low-intensity ones
            if high_pixel_count > low_pixel_count:
                image = 255 - image

        # Convert to PIL image format for saving as PNG
        image_pil = Image.fromarray(image)

        image_pil = image_to_base64(image_pil)

        # image_pil = decode_base64_to_image(image_pil)

        pixel_scale = float(dicom_data.get('PixelSpacing', None)[0])

        json_obj = {'image_base64': image_pil, 'pixel_scale': pixel_scale}

        return jsonify(json_obj)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()

    # Automatically use the image's own format if available
    img_format = img.format if img.format else "PNG"

    img.save(buffered, format=img_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_string):

    img_data = base64.b64decode(base64_string)
    img_pil = Image.open(BytesIO(img_data))
    # Convert PIL image to numpy array
    image_np = np.array(img_pil)

    # Convert RGB to BGR (OpenCV format)
    # image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_pil = Image.fromarray(image_np)

    image_pil.show()

    return image_pil


def getFileFullPath(file_name):
    file_path = "../shared/dicom/" + file_name
    return file_path