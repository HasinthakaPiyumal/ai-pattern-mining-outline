# Cluster 17

def _prepString(string):
    """
    Preprocessing the sentence and splitting the decimal, integer or special
    """
    decimal_set = re.findall('\\d+\\.\\d+', string)
    sub_str = re.sub('\\d+\\.\\d+', '_', string)
    newStr = _replaceDecimal(decimal_set, sub_str)
    integer_set = re.findall('\\d+年', newStr)
    sub_str = re.sub('\\d+年', '_', newStr)
    newStr = _replaceSpecial(integer_set, sub_str)
    integer_set = re.findall('\\d+', newStr)
    sub_str = re.sub('\\d+', '_', newStr)
    newStr = _replaceInteger(integer_set, sub_str)
    print('原句子:', string)
    print('新句子:', newStr)
    print('\n')
    return newStr

def _replaceDecimal(decimal_set, sub_str):
    """ 
    Replacing decimal numbers with Chinese expression
    """
    dec_str_set = []
    for dec in decimal_set:
        dec_str = _float2Chinese(float(dec))
        dec_str_set.append(dec_str)
    newStr = ''
    count = 0
    for c in sub_str:
        if c == '_':
            newStr += dec_str_set[count]
            count += 1
        else:
            newStr += c
    return newStr

def _replaceSpecial(integer_set, sub_str):
    """ 
    Replacing special numbers with Chinese expression, eg: 2018年 --> 二年一八年
    """
    charNumSet = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    int_str_set = []
    for inte in integer_set:
        int_sub_result = ''
        for c in inte:
            if c == '年':
                int_sub_result += '年'
            else:
                int_sub_result += charNumSet[int(c)]
        int_str_set.append(int_sub_result)
    newStr = ''
    count = 0
    for c in sub_str:
        if c == '_':
            newStr += int_str_set[count]
            count += 1
        else:
            newStr += c
    return newStr

def _replaceInteger(integer_set, sub_str):
    """ 
    Replacing integer numbers with Chinese expression
    """
    int_str_set = []
    for inte in integer_set:
        int_str = _integer2Chinese(int(inte))
        int_str_set.append(int_str)
    newStr = ''
    count = 0
    for c in sub_str:
        if c == '_':
            newStr += int_str_set[count]
            count += 1
        else:
            newStr += c
    return newStr

