import pydicom
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import json
import cv2
# import dicom_extraction_latest as dcm
import os
import base64
import sys
from flask import Flask, request


app = Flask(__name__)


@app.route('/extract_image_from_dicom', methods=['POST'])
def extract_image_from_dicom(dicom_file = None):
    """
    Extracts an image from a DICOM file and saves it as a PNG.

    Parameters:
    dicom_file (str): The path to the DICOM file.
    output_file (str): The path to save the extracted image.

    Returns:
    None
    """
    if(dicom_file is None):
        dicom_file = getFileFullPath(request.json['dicom_file'])
    
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_file)

    # Check if the transfer syntax is compressed and handle accordingly
    if 'TransferSyntaxUID' in dicom_data.file_meta and dicom_data.file_meta.TransferSyntaxUID.is_compressed:
        dicom_data.decompress()

    # Get the pixel array from the DICOM data
    pixel_array = dicom_data.pixel_array

    # Apply rescale slope and intercept
    pixel_array = rescale_pixels(pixel_array, dicom_data)

    # Correct photometric interpretation
    pixel_array = correct_photometric_interpretation(pixel_array, dicom_data)

    # Apply windowing to the pixel array
    windowed_array = apply_windowing(pixel_array, dicom_data)

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

    json_obj = json.dumps(json_obj, indent=4)

    return json_obj


@app.route('/process_files', methods=['POST'])
def process_dicom_files():
    worklist_dict = request.json
    # Dictionary to hold the results
    results = {}

    # Temporary dictionary to group scans by patient ID
    patient_scans = {}

    for worklist_id, dicom_path in enumerate(worklist_dict):
        img_str, ds, pixel_scale = dicom_to_base64(getFileFullPath(dicom_path))
        
        patient_id = ds.PatientID
        patient_name = ds.PatientName
        age = ds.PatientAge
        birth_date = ds.PatientBirthDate
        sex = ds.PatientSex
        
        if patient_name not in patient_scans:
            patient_scans[patient_name] = {
                "scan_ids": [],
                "images": [],
                'pixel_scale': [],
                "patient_name": patient_name,
                "age": age,
                "birth_date": birth_date,
                "sex": sex,
                "patient_id": patient_id
            }
        
        patient_scans[patient_name]["scan_ids"].append(worklist_id)
        patient_scans[patient_name]["images"].append(img_str)
        patient_scans[patient_name]["pixel_scale"].append(pixel_scale)

    
    # print(patient_scans[patient_id]["scan_ids"])
    
    # Organize the data as required
    for patient_id, data in patient_scans.items():
        scan_id = data["scan_ids"] #", ".join(str(data["scan_ids"]))
        scan_name = str(data["patient_name"])
        result = {
            'worklist_ids': scan_id,
            "image_1": data["images"][0],
            "image_2": data["images"][1] if len(data["images"]) > 1 else None,
            'pixel_scale_1': data["pixel_scale"][0],
            'pixel_scale_2': data["pixel_scale"][1] if len(data["pixel_scale"]) > 1 else None,
            "patient_name": str(data["patient_name"]),
            "age": data["age"],
            "birth_date": data["birth_date"],
            "sex": data["sex"],
            "patient_id": data["patient_id"]
        }
        
        results[scan_name] = result
    
    return json.dumps(results, indent=4) #results #patient_scans #results #json.dumps(results, indent=4)


def apply_windowing(pixel_array, dicom_data):
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


def rescale_pixels(pixel_array, dicom_data):
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


def correct_photometric_interpretation(pixel_array, dicom_data):
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


def dicom_to_base64(dicom_path):
    # Read the DICOM file
    ds = pydicom.dcmread(dicom_path)
    
    # Extract pixel array and convert to an image
    # pixel_array = ds.pixel_array
    # image = Image.fromarray(pixel_array)

    # image = dcm.extract_image_from_dicom(dicom_path)
    
    # # Convert the image to base64
    # buffered = BytesIO()
    # image.save(buffered, format="PNG")
    # img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    image_obj = extract_image_from_dicom(dicom_path)

    image_obj = json.loads(image_obj)
    
    return image_obj['image_base64'], ds, image_obj['pixel_scale']


def image_to_base64(image_path):

    # Infer the image format from the file extension
    image_format = os.path.splitext(image_path)[-1][1:]
    # print(image_format)

    if image_format.lower() not in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
        raise ValueError("Invalid image format. Choose from 'jpg', 'jpeg', 'png', 'bmp', 'tiff'.")

    image = cv2.imread(image_path)

    _, image_encoded = cv2.imencode(f'.{image_format}', image)

    return base64.b64encode(image_encoded).decode('utf-8')


def getFileFullPath(file_name):
        file_path = "../shared/dicom/" + file_name
        return file_path


if __name__ == "__main__":
    files_list = sys.argv[1]
    result = process_dicom_files(files_list)
