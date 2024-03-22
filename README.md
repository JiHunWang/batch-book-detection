# Batch Book Detection

**The implementation of the batch book detection system - Final project for CS131, Winter 2024**

_Abstract_

_This paper presents an automated system for detecting and cataloging books from images, leveraging computer vi- sion, optical character recognition (OCR), basic NLP tech- niques, and information retrieval. Tested for images un- der various conditions, including different lighting and shelf clutter, the system identifies book titles and fetches related data with an F1 score above 0.8. Notable findings include the effectiveness of noise removal in enhancing OCR ac- curacy in batch book detection. This work offers promis- ing applications for digital library management and per- sonal book-keeping, demonstrating a practical approach to streamline book organization tasks._

This repo contains the code for the batch book detection system.


## Files

- `preprocess.py`: Image pre-processing functions
- `pipeline.py`: The full pipeline from object detection, OCR, NLP, and information retrieval; the default pre-processing technique is the noise removal method, and the regex pattern matching and n-gram analysis are all applied by default
- `utils.py`: A pre-defined list of NLTK stopwords and publishers


## Supported image pre-processing methods
1. Simple channel-averaging grayscale conversion
$$
C = \frac{R + G + B}{3}
$$

2. Luminosity grayscale conversion
$$
C = 0.299R + 0.587G + 0.114B
$$

3. Adaptive thresholding
4. Noise removal
5. Contrast enhancement
6. Thinning and skeletonization


## Supported basic post-processing NLP techniques
1. Remove numbers (e.g. `13`)
2. Remove floats (e.g. `49.78`)
3. Remove publisher names (e.g. `Oxford University Press`)
4. Remove LCC numbers (e.g. `PR4972.6`)
5. Remove independent stopwords (e.g. `this`)
6. Remove symbols (e.g. `+`) 


## Overall pipeline
1. Input image pre-processing: given the variability in image quality, apply pre-processing techniques that can potentially improve the object detection performance
2. Object detection + OCR: utilize the Google Cloud Vision API, which yields a superior performance than YOLOv8 + Tesseract combined
3. Result post-processing: Cloud Vision API detects more candidates than necessary, including false positives. The raw text undergoes a series of post-processing steps to prepare for accurate information retrieval
4. Informatino retrieval: Open Library API linked to the external database is used to retrieve all the relevant book details, e.g. official book titles, author names, etc.


## Dependencies
To set up the dependency, run the following command in the terminal:
```
pip3 install -r requirements.txt
```

## Commands
To see the outputs, it is required to set up one's Google Cloud Vision API set up. The process for this is pretty simple, and it involves enabling the use of API in one's Google Cloud Platform. (This step should not be necessary once the application is deployed.)

Afterwards, run the following command to return all the outputs:
```
python3 pipeline.py -p /PATH/TO/IMAGE
```

## Sample image
I preliminarily provide two sample images, available under the `sample_images/` directory.


## Other information
1. If your image is taken using an iPhone, use the `convert_heic_to_jpeg` method within `preprocess.py` first to convert the image to JPEG before running the command with `python3 pipeline.py`.