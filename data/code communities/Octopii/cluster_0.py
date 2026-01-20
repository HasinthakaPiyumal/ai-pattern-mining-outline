# Cluster 0

def search_pii(file_path):
    contains_faces = 0
    if file_utils.is_image(file_path):
        image = cv2.imread(file_path)
        contains_faces = image_utils.scan_image_for_people(image)
        original, intelligible = image_utils.scan_image_for_text(image)
        text = original
    elif file_utils.is_pdf(file_path):
        pdf_pages = convert_from_path(file_path, 400)
        for page in pdf_pages:
            contains_faces = image_utils.scan_image_for_people(page)
            original, intelligible = image_utils.scan_image_for_text(page)
            text = original
    else:
        text = textract.process(file_path).decode()
        intelligible = text_utils.string_tokenizer(text)
    addresses = text_utils.regional_pii(text)
    emails = text_utils.email_pii(text, rules)
    phone_numbers = text_utils.phone_pii(text, rules)
    keywords_scores = text_utils.keywords_classify_pii(rules, intelligible)
    score = max(keywords_scores.values())
    pii_class = list(keywords_scores.keys())[list(keywords_scores.values()).index(score)]
    country_of_origin = rules[pii_class]['region']
    identifiers = text_utils.id_card_numbers_pii(text, rules)
    if score < 5:
        pii_class = None
    if len(identifiers) != 0:
        identifiers = identifiers[0]['result']
    if temp_dir in file_path:
        file_path = file_path.replace(temp_dir, '')
        file_path = urllib.parse.unquote(file_path)
    result = {'file_path': file_path, 'pii_class': pii_class, 'score': score, 'country_of_origin': country_of_origin, 'faces': contains_faces, 'identifiers': identifiers, 'emails': emails, 'phone_numbers': phone_numbers, 'addresses': addresses}
    return result

def is_image(file_path):
    try:
        i = Image.open(file_path)
        return True
    except:
        return False

def scan_image_for_people(image):
    image = numpy.array(image)
    cascade_values_file = 'face_cascade.xml'
    cascade_values = cv2.CascadeClassifier(cascade_values_file)
    faces = cascade_values.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return len(faces)

def is_pdf(file_path):
    try:
        convert_from_path(file_path, 100)
        return True
    except:
        return False

def regional_pii(text):
    import nltk
    from nltk import word_tokenize, pos_tag, ne_chunk
    from nltk.corpus import stopwords
    resources = ['punkt', 'maxent_ne_chunker', 'stopwords', 'words', 'averaged_perceptron_tagger']
    try:
        nltk_resources = ['tokenizers/punkt', 'chunkers/maxent_ne_chunker', 'corpora/words.zip']
        for resource in nltk_resources:
            if not nltk.data.find(resource):
                raise LookupError()
    except LookupError:
        for resource in resources:
            nltk.download(resource)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    named_entities = ne_chunk(tagged_words)
    locations = []
    for entity in named_entities:
        if isinstance(entity, nltk.tree.Tree):
            if entity.label() in ['GPE', 'GSP', 'LOCATION', 'FACILITY']:
                location_name = ' '.join([word for word, tag in entity.leaves() if word.lower() not in stop_words and len(word) > 2])
                locations.append(location_name)
    return list(set(locations))

def email_pii(text, rules):
    email_rules = rules['Email']['regex']
    email_addresses = re.findall(email_rules, text)
    email_addresses = list(set(filter(None, email_addresses)))
    return email_addresses

def phone_pii(text, rules):
    phone_rules = rules['Phone Number']['regex']
    phone_numbers = re.findall(phone_rules, text)
    phone_numbers = list(itertools.chain(*phone_numbers))
    phone_numbers = list(set(filter(None, phone_numbers)))
    return phone_numbers

def id_card_numbers_pii(text, rules):
    results = []
    regional_regexes = {}
    for key in rules.keys():
        region = rules[key]['region']
        if region is not None:
            regional_regexes[key] = rules[key]
    for key in regional_regexes.keys():
        region = rules[key]['region']
        rule = rules[key]['regex']
        try:
            match = re.findall(rule, text)
        except:
            match = []
        if len(match) > 0:
            result = {'identifier_class': key, 'result': list(set(match))}
            results.append(result)
    return results

