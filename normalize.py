import pydicom

def normalize_ref_image(image):
    #Load DICOM image
    dicom_data = pydicom.dcmread(image)
    #dicom structure:
    #(0028, 1052) Rescale Intercept
    #(0028, 1053) Rescale Slope                
    intercept = dicom_data[(0x0028, 0x1052)].value
    slope = dicom_data[(0x0028, 0x1053)].value
    array_dist = dicom_data.pixel_array
    size = array_dist.shape
    
    if slope is not None and intercept is not None:
        array_dist = (array_dist * slope + intercept)
        return array_dist, size
    else:
        return None, None
    
def normalize_eval_image(image):
    dicom_data = pydicom.dcmread(image)
    intercept = dicom_data[(0x0028, 0x1052)].value
    slope = dicom_data[(0x0028, 0x1053)].value
    array_dist = dicom_data.pixel_array
    size = array_dist.shape
    exposure_sequence = dicom_data[(0x3002, 0x0030)].value
    meterset_exposure = exposure_sequence[0][(0x3002, 0x0032)].value
       
    if slope is not None and intercept is not None and meterset_exposure is not None:
        array_dist = (array_dist * slope + intercept) / meterset_exposure
        return array_dist, size
    else:
        return None, None
    
    
