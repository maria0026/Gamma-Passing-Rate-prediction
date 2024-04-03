import pydicom

def normalize_image(image, reference=True, evaluation=False):
    #Load DICOM image
    dicom_data = pydicom.dcmread(image)
    #dicom structure:
    #(0028, 1052) Rescale Intercept
    #(0028, 1053) Rescale Slope                
    intercept = dicom_data[(0x0028, 0x1052)].value
    slope = dicom_data[(0x0028, 0x1053)].value
    array_dist = dicom_data.pixel_array
    size = array_dist.shape
    
    if reference and slope is not None and intercept is not None:
        array_dist = (array_dist * slope + intercept)
        return array_dist, slope, intercept, size
    
    elif evaluation and slope is not None and intercept is not None:
        exposure_sequence = dicom_data[(0x3002, 0x0030)].value
        meterset_exposure = exposure_sequence[0][(0x3002, 0x0032)].value
        if meterset_exposure is not None:
            array_dist = (array_dist * slope + intercept) / meterset_exposure
            return array_dist, slope, intercept, meterset_exposure, size
        else:
            return array_dist, slope, intercept, None, size
    else:
        return array_dist, None, None, None, size