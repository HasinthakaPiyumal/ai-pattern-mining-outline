# Cluster 85

class TestExtract_emoji(TestCase):

    def test_extract_emoji(self):
        text_ords = [128589, 127998, 97, 102, 100, 115, 128077, 128077, 127998, 127873, 128175]
        text = ''.join([chr(c) for c in text_ords])
        result = [[ord(c) for c in pic] for pic in extract_emoji(text)]
        self.assertEqual(result, [[128589, 127998], [128077], [128077, 127998], [127873], [128175]])

    def test_extract_emoji_ensure_no_numbers(self):
        text_ords = [50, 49, 51, 52, 50, 51, 128587, 127995, 128587, 127995, 97, 32, 97, 106, 97, 107, 115, 100, 108, 32, 102, 97, 115, 108, 107, 51, 32, 107, 32, 51, 32, 35, 32, 94, 32, 64, 32, 33, 32, 32, 35, 32, 42, 32, 60, 32, 62, 32, 63, 32, 32, 34, 32, 46, 32, 44, 32, 32, 41, 32, 40, 32, 36]
        text = ''.join([chr(c) for c in text_ords])
        result = [[ord(c) for c in pic] for pic in extract_emoji(text)]
        self.assertEqual(result, [[128587, 127995], [128587, 127995]])

def extract_emoji(text):
    """
	Parameters
	----------
	text, str

	Returns
	-------
	List of 5.0-compliant emojis that occur in text.
	"""
    found_emojis = []
    len_text = len(text)
    i = 0
    while i < len_text:
        cur_char = ord(text[i])
        try:
            VALID_EMOJIS[cur_char]
        except:
            i += 1
            continue
        found = False
        for dict_len, candidates in VALID_EMOJIS[cur_char]:
            if i + dict_len <= len_text:
                if dict_len == 0:
                    _append_if_valid(found_emojis, text[i])
                    i += 1
                    found = True
                    break
                candidate = tuple((ord(c) for c in text[i + 1:i + 1 + dict_len]))
                if candidate in candidates:
                    _append_if_valid(found_emojis, text[i:i + 1 + dict_len])
                    i += 1 + dict_len
                    found = True
                    break
            if found:
                break
        if not found:
            _append_if_valid(found_emojis, text[i])
            i += 1
    return found_emojis

def _append_if_valid(found_emojis, candidate):
    for c in candidate:
        if ord(c) > 1000:
            found_emojis.append(candidate)
            return

class FeatsFromSpacyDocOnlyEmoji(FeatsFromSpacyDoc):
    """
	Strips away everything but emoji tokens from spaCy
	"""

    def get_feats(self, doc):
        """
		Parameters
		----------
		doc, Spacy Docs

		Returns
		-------
		Counter emoji -> count
		"""
        return Counter(extract_emoji(str(doc)))

