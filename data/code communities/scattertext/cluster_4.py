# Cluster 4

def train_sentence_piece_tokenizer(documents, vocab_size):
    """
    :param documents: list-like, a list of str documents
    :vocab_size int: the size of the vocabulary to output

    :return sentencepiece.SentencePieceProcessor
    """
    sp = None
    with tempfile.NamedTemporaryFile(delete=True) as tempf:
        with tempfile.NamedTemporaryFile(delete=True) as tempm:
            tempf.write('\n'.join(documents).encode())
            mod = spm.SentencePieceTrainer.Train('--input=%s --model_prefix=%s --vocab_size=%s' % (tempf.name, tempm.name, vocab_size))
            sp = spm.SentencePieceProcessor()
            sp.load(tempm.name + '.model')
    return sp

