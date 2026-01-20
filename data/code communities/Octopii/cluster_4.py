# Cluster 4

def scan_image_for_text(image):
    image = numpy.array(image)
    try:
        image_text_unmodified = pytesseract.image_to_string(image, config='--psm 12')
    except TypeError:
        print('Cannot open this file type.')
        return
    try:
        try:
            degrees_to_rotate = pytesseract.image_to_osd(image)
        except:
            degrees_to_rotate = 'Rotate: 180'
        for item in degrees_to_rotate.split('\n'):
            if 'rotate'.lower() in item.lower():
                degrees_to_rotate = int(item.replace(' ', '').split(':', 1)[1])
                if degrees_to_rotate == 180:
                    pass
                elif degrees_to_rotate == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif degrees_to_rotate == 360:
                    image = cv2.rotate(image, cv2.ROTATE_180)
        image_text_auto_rotate = pytesseract.image_to_string(image, config='--psm 12')
    except:
        print("Couldn't auto-rotate image")
        image_text_auto_rotate = ''
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_text_grayscaled = pytesseract.image_to_string(image, config='--psm 12')
    except:
        print("Couldn't grayscale image")
        image_text_grayscaled = ''
    try:
        image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        image_text_monochromed = pytesseract.image_to_string(image, config='--psm 12')
    except:
        print("Couldn't monochrome image")
        image_text_monochromed = ''
    try:
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        image_text_mean_threshold = pytesseract.image_to_string(image, config='--psm 12')
    except:
        print("Couldn't mean threshold image")
        image_text_mean_threshold = ''
    try:
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        image_text_gaussian_threshold = pytesseract.image_to_string(image, config='--psm 12')
    except:
        print("Couldn't gaussian threshold image")
        image_text_gaussian_threshold = ''
    try:
        angle = determine_skew(image)
        rotated = rotate(image, angle, resize=True) * 255
        image = rotated.astype(numpy.uint8)
        image_text_deskewed_1 = pytesseract.image_to_string(image, config='--psm 12')
        angle = determine_skew(image)
        rotated = rotate(image, angle, resize=True) * 255
        image = rotated.astype(numpy.uint8)
        image_text_deskewed_2 = pytesseract.image_to_string(image, config='--psm 12')
        angle = determine_skew(image)
        rotated = rotate(image, angle, resize=True) * 255
        image = rotated.astype(numpy.uint8)
        image_text_deskewed_3 = pytesseract.image_to_string(image, config='--psm 12')
    except:
        print("Couldn't deskew image")
        image_text_deskewed_1 = ''
        image_text_deskewed_2 = ''
        image_text_deskewed_3 = ''
    unmodified_words = text_utils.string_tokenizer(image_text_unmodified)
    grayscaled = text_utils.string_tokenizer(image_text_grayscaled)
    auto_rotate = text_utils.string_tokenizer(image_text_auto_rotate)
    monochromed = text_utils.string_tokenizer(image_text_monochromed)
    mean_threshold = text_utils.string_tokenizer(image_text_mean_threshold)
    gaussian_threshold = text_utils.string_tokenizer(image_text_gaussian_threshold)
    deskewed_1 = text_utils.string_tokenizer(image_text_deskewed_1)
    deskewed_2 = text_utils.string_tokenizer(image_text_deskewed_2)
    deskewed_3 = text_utils.string_tokenizer(image_text_deskewed_3)
    original = image_text_unmodified + '\n' + image_text_auto_rotate + '\n' + image_text_grayscaled + '\n' + image_text_monochromed + '\n' + image_text_mean_threshold + '\n' + image_text_gaussian_threshold + '\n' + image_text_deskewed_1 + '\n' + image_text_deskewed_2 + '\n' + image_text_deskewed_3
    intelligible = unmodified_words + grayscaled + auto_rotate + monochromed + mean_threshold + gaussian_threshold + deskewed_1 + deskewed_2 + deskewed_3
    return (original, intelligible)

def string_tokenizer(text):
    final_word_list = []
    words_list = text.replace(' ', '\n').split('\n')
    for element in words_list:
        if len(element) >= 2:
            final_word_list.append(element)
    return final_word_list

