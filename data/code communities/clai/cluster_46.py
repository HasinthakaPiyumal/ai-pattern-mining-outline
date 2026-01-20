# Cluster 46

def identify_sections(sentences: list):
    sentences = [cleanup(sentence) for sentence in sentences]
    segments = []
    start_index = 0
    empty_sentence = 0
    for idx, sentence in enumerate(sentences):
        if not sentence:
            if empty_sentence >= 1:
                segments.append(sentences[start_index:idx])
                start_index = idx + 1
                empty_sentence = 0
            else:
                empty_sentence += 1
    sections = list()
    for segment in segments:
        sentences = list(filter(None, segment))
        if sentences:
            sections.append(' '.join([sentences for sentences in segment if sentences]))
    return sections

def cleanup(text: str) -> str:
    text = PATTERN_SPACES.sub(' ', text).strip()
    text = PATTERN_DASH.sub('', text).strip()
    return text

def parse_manpage(text: list) -> str:
    return ' '.join(identify_sections(text))

def load_manpages(print_every=1000.0) -> dict:
    logging.info('==================================')
    logging.info('Processing manpages')
    logging.info('==================================')
    documents = {}
    files = pathlib.Path('./data/manpages/').glob('*.txt')
    for idx, f in enumerate(files):
        with open(f, 'r') as fp:
            content = parse_manpage(fp.readlines())
            if content:
                documents[f.name.split('.txt')[0]] = content
        if idx % print_every == 0:
            msg = 'Finished processing {} manpages.'.format(idx)
            logging.info(msg)
    return documents

def save(func, vectors, keys):
    with open(absolute_path('./data/model/func.p'), 'wb') as fp:
        pickle.dump(func, fp)
    with open(absolute_path('./data/model/vectors.p'), 'wb') as fp:
        pickle.dump(vectors, fp)
    with open(absolute_path('./data/model/keys.p'), 'wb') as fp:
        pickle.dump(keys, fp)

def absolute_path(relative_path):
    return os.path.join(_BASE_PATH, relative_path)

def transform(corpus) -> (TfidfVectorizer, list):
    vectorizer = TfidfVectorizer(input='content', lowercase=True, analyzer='word', stop_words='english', ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(corpus)
    vectorizer.stop_words_ = None
    return (vectorizer, vectors)

