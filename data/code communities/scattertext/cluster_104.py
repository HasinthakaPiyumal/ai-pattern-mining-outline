# Cluster 104

def extract_finditer(pos_seq, regex=SimpleNP):
    """The "GreedyFSA" method in Handler et al. 2016.
	Returns token position spans of valid ngrams."""
    ss = coarse_tag_str(pos_seq)

    def gen():
        for m in re.finditer(regex, ss):
            yield (m.start(), m.end())
    return list(gen())

def coarse_tag_str(pos_seq):
    """Convert POS sequence to our coarse system, formatted as a string."""
    global tag2coarse
    tags = [tag2coarse.get(tag, 'O') for tag in pos_seq]
    return ''.join(tags)

def gen():
    for s in xrange(len(ss)):
        for n in xrange(minlen, 1 + min(maxlen, len(ss) - s)):
            e = s + n
            substr = ss[s:e]
            if re.match(regex + '$', substr):
                yield (s, e)

def extract_ngram_filter(pos_seq, regex=SimpleNP, minlen=1, maxlen=8):
    """The "FilterFSA" method in Handler et al. 2016.
	Returns token position spans of valid ngrams."""
    ss = coarse_tag_str(pos_seq)

    def gen():
        for s in xrange(len(ss)):
            for n in xrange(minlen, 1 + min(maxlen, len(ss) - s)):
                e = s + n
                substr = ss[s:e]
                if re.match(regex + '$', substr):
                    yield (s, e)
    return list(gen())

class SpacyTagger:

    def __init__(self):
        self.spacy_object = None

    def tag_text(self, text):
        text = unicodify(text)
        doc = self.spacy_object(text)
        return {'pos': [token.tag_ for token in doc], 'tokens': [token.text for token in doc]}

    def tag_tokens(self, tokens):
        newtext = safejoin(tokens)
        newtext = unicodify(newtext)
        return self.tag_text(newtext)

def unicodify(s, encoding='utf8', errors='ignore'):
    if sys.version_info[0] < 3:
        if isinstance(s, unicode):
            return s
        if isinstance(s, str):
            return s.decode(encoding, errors)
        return unicode(s)
    elif type(s) == bytes:
        return s.decode('utf8')
    else:
        return s

def safejoin(list_of_str_or_unicode):
    xx = list_of_str_or_unicode
    if not xx:
        return u''
    if isinstance(xx[0], str):
        return ' '.join(xx)
    if isinstance(xx[0], bytes):
        return ' '.join(xx)
    if sys.version_info[0] < 3:
        if isinstance(xx[0], unicode):
            return u' '.join(xx)
    raise Exception('Bad input to safejoin:', list_of_str_or_unicode)

def get_phrases(text=None, tokens=None, postags=None, tagger='nltk', grammar='SimpleNP', regex=None, minlen=2, maxlen=8, output='counts'):
    """Give a text (or POS tag sequence), return the phrases matching the given
	grammar.  Works on documents or sentences.
	Returns a dict with one or more keys with the phrase information.

	text: the text of the document.  If supplied, we will try to POS tag it.

	You can also do your own tokenzation and/or tagging and supply them as
	'tokens' and/or 'postags', which are lists of strings (of the same length).
	 - Must supply both to get phrase counts back.
	 - With only postags, can get phrase token spans back.
	 - With only tokens, we will try to POS-tag them if possible.

	output: a string, or list of strings, of information to return. Options include:
	 - counts: a Counter with phrase frequencies.  (default)
	 - token_spans: a list of the token spans of each matched phrase.  This is
		 a list of (start,end) pairs of integers, which refer to token positions.
	 - pos, tokens can be returned too.

	tagger: if you're passing in raw text, can supply your own tagger, from one
	of the get_*_tagger() functions.  If this is not supplied, we will try to load one.

	grammar: the grammar to use.  Only one option right now...

	regex: a custom regex to use, instead of a premade grammar.  Currently,
	this must work on the 5-tag system described near the top of this file.

	"""
    global SimpleNP
    if postags is None:
        try:
            tagger = TAGGER_NAMES[tagger]()
        except:
            raise Exception("We don't support tagger %s" % tagger)
        d = None
        if tokens is not None:
            d = tagger.tag_tokens(tokens)
        elif text is not None:
            d = tagger.tag_text(text)
        else:
            raise Exception('Need to supply text or tokens.')
        postags = d['pos']
        tokens = d['tokens']
    if regex is None:
        if grammar == 'SimpleNP':
            regex = SimpleNP
        else:
            assert False, "Don't know grammar %s" % grammar
    phrase_tokspans = extract_ngram_filter(postags, minlen=minlen, maxlen=maxlen)
    if isinstance(output, str):
        output = [output]
    our_options = set()

    def retopt(x):
        our_options.add(x)
        return x in output
    ret = {}
    ret['num_tokens'] = len(postags)
    if retopt('token_spans'):
        ret['token_spans'] = phrase_tokspans
    if retopt('counts'):
        counts = Counter()
        for start, end in phrase_tokspans:
            phrase = safejoin([tokens[i] for i in xrange(start, end)])
            phrase = phrase.lower()
            counts[phrase] += 1
        ret['counts'] = counts
    if retopt('pos'):
        ret['pos'] = postags
    if retopt('tokens'):
        ret['tokens'] = tokens
    xx = set(output) - our_options
    if xx:
        raise Exception("Don't know how to handle output options: %s" % list(xx))
    return ret

def retopt(x):
    our_options.add(x)
    return x in output

