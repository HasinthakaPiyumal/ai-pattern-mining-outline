# Cluster 2

def convertDigit2Character(string):
    return _prepString(string)

def convertCharacter2Digit(string):
    chinese_numbers = re.findall(u'[点零一二三四五六七八九十百千万亿]{1,}', string, re.S)
    sub_str = re.sub(u'[点零一二三四五六七八九十百千万亿]{1,}', '_', string)
    for chinese_number in chinese_numbers:
        digit = _convert_all(chinese_number)
        sub_str = sub_str.replace('_', digit, 1)
    print('原句子:', string)
    print('新句子:', sub_str)
    print('\n')
    return sub_str

class DigitPrecessor(object):

    def __init__(self, mode):
        assert mode == 'digit2char' or mode == 'char2digit', 'Wrong mode: %s' % str(mode)
        self.mode = mode

    def processString(self, string):
        if self.mode == 'digit2char':
            return convertDigit2Character(string)
        else:
            return convertCharacter2Digit(string)

    def processFile(self, fileName):
        result = []
        assert os.path.isfile(fileName), 'Wrong file path: %s' % str(fileName)
        with codecs.open(fileName, 'r', 'utf-8') as f:
            content = f.readlines()
        if self.mode == 'digit2char':
            for string in content:
                result.append(convertDigit2Character(string))
        else:
            for string in content:
                result.append(convertCharacter2Digit(string))
        return result

