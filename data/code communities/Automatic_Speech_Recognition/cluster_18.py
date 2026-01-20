# Cluster 18

def _integer2Chinese(number):
    """
    Converting integer to Chinese expression
    """
    charSectionSet = ['', '万', '亿', '万亿']
    result = ''
    zero = False
    unitPos = 0
    if number == 0:
        return '零'
    while number > 0:
        section = number % 10000
        if zero:
            result = '零' + result
        sec_result = _section2Chinese(section)
        if section != 0:
            sec_result += charSectionSet[unitPos]
        result = sec_result + result
        if section < 1000 and section > 0:
            zero = True
        number = number // 10000
        unitPos += 1
    return result

def _section2Chinese(section):
    """
    Converting section to Chinese expression
    """
    result = ''
    charNumSet = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    charUnitSet = ['', '十', '百', '千']
    zero = True
    unitPos = 0
    while section > 0:
        v = section % 10
        if v == 0:
            if section == 0 or zero is False:
                zero = True
                result = charNumSet[v] + result
        elif section // 10 == 0 and v == 1 and (unitPos == 1):
            result = charUnitSet[1] + result
        else:
            zero = False
            strIns = charNumSet[v]
            strIns += charUnitSet[unitPos]
            result = strIns + result
        unitPos += 1
        section = section // 10
    return result

def _float2Chinese(number):
    """
    Converting floating number to Chinese expression
    """
    charNumSet = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    integer_part, decimal_part = str(number).split('.')
    int_result = _integer2Chinese(int(integer_part))
    dec_result = ''
    for c in decimal_part:
        dec_result += charNumSet[int(c)]
    return int_result + '点' + dec_result

