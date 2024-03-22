import argparse
import grequests
import requests
import re

from utils import publishers, stopwords
from preprocess import (
    convert_grayscale_simple,
    convert_grayscale_luminosity,
    otsu_binarization,
    adaptive_thresholding,
    skew_correction,
    noise_removal,
    thinning_skeletonization,
    enhanced_binarization
)


def is_number(str):
    """Check if the detected text is a number."""
    try:
        int(str)
        return True
    except ValueError:
        return False
    

def is_float(str):
    """Check if the detected text is a floating number."""
    float_regex = r'^\s*[+-]?(\d+\.\d*|\.\d+)\s*$'
    return re.match(float_regex, str) is not None


def is_lcc_number(str):
    """Check if the detected text is an instance of LCC number."""
    pattern_1 = r'^[A-Z]+\d+\.\d+$'  # example: PR9369.4
    if bool(re.match(pattern_1, str)):
        return True
    pattern_2 = r'^[A-Z]+\d+$'  # example: T74
    if bool(re.match(pattern_2, str)):
        return True
    pattern_3 = r'^\.[A-Z]+\d+$'  # example: .V744
    if bool(re.match(pattern_3, str)):
        return True
    return False


def is_publisher(str):
    """Check if the detected text is a publisher."""
    str = str.lower()
    if str in publishers:
        return True
    return False


def is_stopword(str):
    """Check if the detected text is a stopword."""
    str = str.lower()
    if str in stopwords:
        return True
    return False


def is_single_alpha_character(str):
    """Check if the detected text consists of pure English alphabets with a length at most 2."""
    return len(str) <= 2 and str.isalpha()


def is_symbol(str):
    """Check if the detected text is a symbol."""
    if str in ['•', ',', '.', '+', '-', '*', '/']:
        return True
    return False


def is_etc(str):
    """Check if the detected text belongs to the list of other miscellaneous strings."""
    if str in ['ps']:
        return True
    return False


def detect_text(path):
    """Detects text in the image.
        First, send a request to the Google Cloud Vision API.
        Second, send a GET request to the Open Library API.
        Return 
    """
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()
        content = noise_removal(content)

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    title_candidates = set()
    all_book_infos = set()

    for text in texts:
        print(f'\n"{text.description}"')


    if texts:
        lines = texts[0].description.split('\n')
        for line in lines:
            if (not is_float(line)) and (not is_lcc_number(line)) \
                and (not is_number(line)) and (not is_publisher(line)) \
                and (not is_stopword(line)) and (not is_single_alpha_character(line)) \
                and (not is_symbol(line)) and (not is_etc(line)):
                print(f'adding: {line}')
                title_candidates.add(line)
        
        urls = [construct_endpoint_url(title_candidate) for title_candidate in title_candidates]
        unsent_requests = (grequests.get(url) for url in urls)
        responses = grequests.map(unsent_requests)
        for response in responses:
            if response:
                r = response.json()
                if len(r['docs']) != 0:
                    title = r['docs'][0]['title']
                    author = r['docs'][0]['author_name'][0]
                    result = {'title': title, 'author': author}
                    all_book_infos.add(tuple(result.items()))

    else:
        return None


    # if response.error.message:
    #     raise Exception(
    #         "{}\nFor more info on error messages, check: "
    #         "https://cloud.google.com/apis/design/errors".format(response.error.message)
    #     )
    
    return all_book_infos


def construct_endpoint_url(title):
    """Construct an endpoint URL to send a GET request to Open Library API."""
    return 'https://openlibrary.org/search.json?q=' + '+'.join(title.split())


def fetch_book_info_non_async(title):
    """A regular version of book fetch (the async one is preferred)."""
    url = 'https://openlibrary.org/search.json?q=' + '+'.join(title.split())
    print('\n' * 2, 'title:', title)
    response = requests.get(url)
    print('response:', response)
    r = response.json()
    if len(r['docs']) != 0:
        title = r['docs'][0]['title']
        author = r['docs'][0]['author_name'][0]
        return {'title': title, 'author': author}
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path')
    args = parser.parse_args()
    result = detect_text(args.path)
    print(result)