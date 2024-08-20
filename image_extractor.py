import os
import pydicom
from PIL import Image
import numpy as np
from io import BytesIO
# from IPython.display import display
import base64
import json
import sys
from flask_restful import Resource
from flask import jsonify


class ImageExtractor(Resource):
    def get(self, file_name):
        return self.extract_image_from_dicom(file_name)

    
    def apply_windowing(self, pixel_array, dicom_data):
        """
        Apply windowing to the pixel array using the DICOM Window Center and Window Width.

        Parameters:
        pixel_array (numpy.ndarray): The pixel array from the DICOM file.
        dicom_data (pydicom.Dataset): The DICOM dataset.

        Returns:
        numpy.ndarray: The windowed pixel array.
        """
        if 'WindowCenter' in dicom_data and 'WindowWidth' in dicom_data:
            window_center = dicom_data.WindowCenter
            window_width = dicom_data.WindowWidth

            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]

            lower_bound = window_center - (window_width / 2)
            upper_bound = window_center + (window_width / 2)

            windowed_array = np.clip(pixel_array, lower_bound, upper_bound)
            windowed_array = (windowed_array - lower_bound) / (upper_bound - lower_bound) * 255
        else:
            # Default normalization if windowing information is not available
            windowed_array = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))) * 255

        return windowed_array.astype(np.uint8)


    def rescale_pixels(self, pixel_array, dicom_data):
        """
        Rescale the pixel array using Rescale Slope and Rescale Intercept if present.

        Parameters:
        pixel_array (numpy.ndarray): The pixel array from the DICOM file.
        dicom_data (pydicom.Dataset): The DICOM dataset.

        Returns:
        numpy.ndarray: The rescaled pixel array.
        """
        if 'RescaleSlope' in dicom_data and 'RescaleIntercept' in dicom_data:
            slope = dicom_data.RescaleSlope
            intercept = dicom_data.RescaleIntercept
            pixel_array = pixel_array * slope + intercept
        return pixel_array


    def correct_photometric_interpretation(self, pixel_array, dicom_data):
        """
        Correct the photometric interpretation of the pixel array.

        Parameters:
        pixel_array (numpy.ndarray): The pixel array from the DICOM file.
        dicom_data (pydicom.Dataset): The DICOM dataset.

        Returns:
        numpy.ndarray: The corrected pixel array.
        """
        if dicom_data.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = np.max(pixel_array) - pixel_array
        return pixel_array


    def extract_image_from_dicom(self, dicom_file):
        """
        Extracts an image from a DICOM file and saves it as a PNG.

        Parameters:
        dicom_file (str): The path to the DICOM file.
        output_file (str): The path to save the extracted image.

        Returns:
        None
        """
        # Read the DICOM file
        dicom_data = pydicom.dcmread(self.getFileFullPath(dicom_file))

        # Check if the transfer syntax is compressed and handle accordingly
        if 'TransferSyntaxUID' in dicom_data.file_meta and dicom_data.file_meta.TransferSyntaxUID.is_compressed:
            dicom_data.decompress()

        # Get the pixel array from the DICOM data
        pixel_array = dicom_data.pixel_array

        # Apply rescale slope and intercept
        pixel_array = self.rescale_pixels(pixel_array, dicom_data)

        # Correct photometric interpretation
        pixel_array = self.correct_photometric_interpretation(pixel_array, dicom_data)

        # Apply windowing to the pixel array
        windowed_array = self.apply_windowing(pixel_array, dicom_data)

        # Ensure the pixel values are in the correct range
        windowed_array = np.clip(windowed_array, 0, 255)

        # Convert the numpy array to a PIL image
        image = Image.fromarray(windowed_array)

        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        pixel_scale = float(dicom_data.get('PixelSpacing', None)[0])

        json_obj = {'image_base64': img_str, 'pixel_scale': pixel_scale}

        return jsonify(json_obj)


    def process_dicom_folder(self, source_folder, destination_folder, format):
        """
        Process all DICOM files in the source folder and save extracted images to the destination folder.

        Parameters:
        source_folder (str): The path to the source folder containing DICOM files.
        destination_folder (str): The path to the destination folder to save extracted images.

        Returns:
        None
        """
        # Ensure the destination folder exists
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Process each DICOM file in the source folder
        for filename in os.listdir(source_folder):
            if filename.endswith(format):
                dicom_file = os.path.join(source_folder, filename)
                output_file = os.path.join(destination_folder, f"{os.path.splitext(filename)[0]}.png")
                self.extract_image_from_dicom(dicom_file, output_file)
                print(f"Extracted image from {filename} and saved as {output_file}")


    def getFileFullPath(self, file_name):
        file_path = "../shared/dicom/" + file_name
        return file_path