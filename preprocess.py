from PIL import Image
from pillow_heif import register_heif_opener
import cv2
from ultralytics import YOLO
import pyheif
import pytesseract
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
from skimage.color import rgb2gray


# set pytesseract.pytesseract.tesseract_cmd properly
# set `os.environ['TESSDATA_PREFIX']` to something akin to `/usr/local/Cellar/tesseract/VERSION/lang/`
# may need to run `register_heif_opener()`


def convert_heic_to_jpeg(heic_path, output_path):
    """Convert HEIC images to JPEG (images taken with iPhone cameras)."""
    heif_file = pyheif.read(heic_path)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    image.save(output_path, "JPEG")


def load_model(pt_name):
    model = YOLO(pt_name)
    return model


def predict(model, source):
    image = Image.open(source)
    results = model.predict(source=image, save=True)
    return results


def run_ocr(img_path):
    image_tensor = Image.open(img_path)
    result = pytesseract.image_to_string(image_tensor, lang='eng')
    print(result)


def convert_grayscale_simple(img):
    img = np.array(img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def convert_grayscale_luminosity(img, save_path):
    img = np.array(img)
    if len(img.shape) == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        img = np.clip(img, 0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    io.imsave(save_path, img)
    return img


# https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7
def otsu_binarization(img):
    # gives a threshold for the whole image considering the various characteristics
    # e.g. lighting conditions, contrast, sharpness, etc.
    # typically applied to grayscale image
    # img: PIL image (needed to be converted to a NumPy array)
    img = np.array(img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # imgf contains binary image
    print(f"Otsu's threshold: {ret}")
    plt.imshow(imgf)
    plt.axis('off')
    plt.show()


def adaptive_thresholding(img, save_path):
    # gives a threshold for a small part of the image depending on the characteristics
    # of its locality and neighbors (useful when a very small part of the image has
    # a different threshold depending upon the locality)
    img = np.array(img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    imgf = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2) # imgf contains Binary image
    plt.imshow(imgf)
    plt.axis('off')
    plt.show()
    io.imsave(save_path, imgf)


from scipy.ndimage import interpolation as inter
def skew_correction(img):

    # convert to binary
    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
    plt.imshow(bin_img, cmap='gray')

    # projection profile method
    def find_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score
    
    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print(f'Best angle: {best_angle}')
    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img = Image.fromarray((255 * data).astype("uint8")).convert("RGB")
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def noise_removal(img, save_path, grayscale=False):
    img = np.array(img)
    
    if grayscale:
        raise Exception("should be RGB")
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
    # Plotting of source and destination image 
    plt.subplot(121), plt.imshow(img) 
    plt.subplot(122), plt.imshow(dst) 
    plt.show()
    io.imsave(save_path, dst)


def thinning_skeletonization(img, save_path, grayscale=True):
    img = np.array(img)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    plt.imshow(erosion)
    plt.axis('off')
    plt.show()
    io.imsave(save_path, erosion)


def enhanced_binarization(img, save_path):
    # Convert PIL image to a NumPy array
    img = np.array(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    enhanced_gray = clahe.apply(gray)
    
    # Apply Otsu's binarization on the enhanced image
    ret, imgf = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Display the binary image
    plt.imshow(imgf, cmap='gray')
    plt.axis('off')
    plt.show()
    io.imsave(save_path, imgf)


def preprocess_for_ocr(img):
    img_array = np.array(img)
    if len(img_array.shape) == 3:
        img_array = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)
    return img

def extract_text(img, lang='eng'):
    config = '--oem 3 --psm 3'
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    return text

def detect(model, image, lang='eng'):
    img = Image.open(image)
    results = model(img)
    titles = []

    for *box, conf, cls in results[0].boxes.data:
        if model.names[int(cls)] == 'book':
            book_img = img.crop((box[0].item(), box[1].item(), box[2].item(), box[3].item()))
            book_img = preprocess_for_ocr(book_img)
            title = extract_text(book_img, lang=lang)
            titles.append(title.strip())
    return titles
