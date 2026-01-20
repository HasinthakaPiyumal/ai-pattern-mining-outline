# Cluster 9

def normalize_matrix(matrix_str: str) -> str:
    """Helper function to normalize matrices and vectors."""
    logger.debug(f'Normalizing matrix input: {repr(matrix_str)}')
    try:
        matrix_str = ''.join(matrix_str.split())
        match = re.match('^\\\\begin\\{pmatrix\\}(.*?)\\\\end\\{pmatrix\\}$', matrix_str)
        if not match:
            return matrix_str
        content = match.group(1)
        rows = content.split('\\\\')
        normalized_rows = []
        for row in rows:
            if '&' in row:
                entries = [normalize_matrix_entry(entry) for entry in row.split('&')]
            else:
                entries = [normalize_matrix_entry(row)]
            normalized_rows.append('&'.join(entries))
        result = '\\begin{pmatrix}' + '\\\\'.join(normalized_rows) + '\\end{pmatrix}'
        logger.debug(f'Normalized matrix result: {repr(result)}')
        return result
    except Exception as e:
        logger.debug(f'Failed to normalize matrix: {str(e)}')
        return matrix_str

def normalize_matrix_entry(entry: str) -> str:
    """Helper function to normalize a single matrix entry."""
    logger.debug(f'Normalizing matrix entry input: {repr(entry)}')
    entry = ''.join(entry.split())
    if '/' in entry and (not any((c in entry for c in '\\{}'))):
        if entry.startswith('-'):
            num, den = entry[1:].split('/')
            return f'-{num.strip()}/{den.strip()}'
        else:
            num, den = entry.split('/')
            return f'{num.strip()}/{den.strip()}'
    entry = entry.replace('\\dfrac', '\\frac')
    frac_match = re.match('^(-)?\\\\frac\\{(\\d+)\\}\\{(\\d+)\\}$', entry)
    if frac_match:
        sign, num, den = frac_match.groups()
        sign = sign if sign else ''
        return f'{sign}{num}/{den}'
    return entry

def normalize_algebraic_expression(expr: str) -> str:
    """Helper function to normalize algebraic expressions."""
    logger.debug(f'Normalizing algebraic expression: {repr(expr)}')
    try:
        expr = ''.join(expr.split())
        monomial_match = re.match('^(-?\\d*\\.?\\d*)?([a-zA-Z])(?:\\^(-?\\d+))?$', expr)
        if monomial_match:
            coeff, var, exp = monomial_match.groups()
            coeff = coeff if coeff and coeff not in ['+', '-'] else '1' if not coeff else '-1'
            exp = exp if exp else '1'
            if coeff == '1' and exp == '1':
                result = var
            elif coeff == '1':
                result = f'{var}^{exp}'
            elif coeff == '-1' and exp == '1':
                result = f'-{var}'
            elif coeff == '-1':
                result = f'-{var}^{exp}'
            elif exp == '1':
                result = f'{coeff}{var}'
            else:
                result = f'{coeff}{var}^{exp}'
            logger.debug(f'Matched as monomial with exponent: {repr(result)}')
            return result.lower()
        pi_term_match = re.match('^(-?\\d*\\.?\\d*)\\\\?pi$', expr)
        if pi_term_match:
            coeff = pi_term_match.group(1)
            if not coeff or coeff == '-':
                coeff = '-1' if coeff == '-' else '1'
            return f'{coeff}\\pi'
        frac_pi_match = re.match('^\\\\frac{([^{}]+)}{([^{}]+)}\\\\?pi$', expr)
        if frac_pi_match:
            num, den = frac_pi_match.groups()
            return f'\\frac{{{num}}}{{{den}}}\\pi'
        frac_match = re.match('^\\\\frac{([^{}]+)}{([^{}]+)}$', expr)
        if frac_match:
            num, den = frac_match.groups()
            return f'\\frac{{{num}}}{{{den}}}'
        terms = []
        current_term = ''
        for i, char in enumerate(expr):
            if char in ['+', '-'] and i > 0:
                if current_term:
                    terms.append(current_term)
                current_term = char
            else:
                current_term += char
        if current_term:
            terms.append(current_term)
        if len(terms) == 1 and re.match('^-?[\\d,]+$', terms[0]):
            return normalize_number(terms[0])
        processed_terms = []
        for term in terms:
            if term.startswith('+'):
                term = term[1:]
            if not term.startswith('-'):
                term = '+' + term
            match = re.match('^([+-])?\\s*(\\d*\\.?\\d*)?([a-zA-Z](?:\\^\\d+)?)?$', term)
            if match:
                sign, coeff, var = match.groups()
                if not coeff and var:
                    coeff = '1'
                elif not coeff:
                    coeff = '0'
                processed_terms.append((sign, float(coeff), var or ''))
        processed_terms.sort(key=lambda x: (not bool(x[2]), x[2], -x[1]))
        result = ''
        for sign, coeff, var in processed_terms:
            if coeff == 0:
                continue
            term = ''
            if coeff == 1 and var:
                term = var
            elif coeff == -1 and var:
                term = f'-{var}'
            elif var:
                term = f'{coeff}{var}'
            else:
                term = str(coeff)
            if result and term[0] != '-':
                result += '+'
            result += term
        logger.debug(f'Normalized algebraic expression result: {repr(result)}')
        return result.lower()
    except Exception as e:
        logger.debug(f'Failed to normalize algebraic expression: {str(e)}')
        return expr.lower()

def normalize_number(num_str: str) -> str:
    """Helper function to normalize number representation."""
    try:
        cleaned = re.sub('[,\\$\\\\]|\\s*(?:cm|m|kg|ft|in|lb|oz|ml|L)$|\\s*\\\\text{[^}]+}', '', num_str).strip()
        if cleaned.startswith('.'):
            cleaned = '0' + cleaned
        num = float(cleaned)
        if abs(num) < 1 and '.' in cleaned:
            decimal_places = len(cleaned.split('.')[1])
            format_str = f'{{:.{decimal_places}f}}'
            result = format_str.format(num)
        else:
            result = str(num)
        logger.debug(f'Normalized number result: {repr(result)}')
        return result
    except Exception as e:
        logger.debug(f'Failed to normalize number: {str(e)}')
        return num_str

def normalize_interval(interval_str: str) -> str:
    """Helper function to normalize intervals."""
    logger.debug(f'Normalizing interval: {repr(interval_str)}')
    try:
        interval_str = ''.join(interval_str.split())
        match = re.match('^\\\\left?([\\[\\(])(.*?),(.*?)\\\\right?([\\]\\)])$', interval_str)
        if not match:
            match = re.match('^([\\[\\(])(.*?),(.*?)([\\]\\)])$', interval_str)
            if not match:
                return interval_str
        left_bracket, left_bound, right_bound, right_bracket = match.groups()
        norm_left = normalize_interval_bound(left_bound)
        norm_right = normalize_interval_bound(right_bound)
        result = f'\\left{left_bracket}{norm_left},{norm_right}\\right{right_bracket}'
        logger.debug(f'Normalized interval result: {repr(result)}')
        return result
    except Exception as e:
        logger.debug(f'Failed to normalize interval: {str(e)}')
        return interval_str

def normalize_interval_bound(bound: str) -> str:
    """Helper function to normalize interval bounds."""
    logger.debug(f'Normalizing interval bound: {repr(bound)}')
    if '\\infty' in bound:
        sign = '-' if bound.startswith('-') else ''
        return f'{sign}\\infty'
    return normalize_answer(bound) or bound

def normalize_answer(answer: str) -> str:
    """Normalize the answer string for comparison."""
    logger.debug(f'Normalizing answer: {repr(answer)}')
    if answer is None:
        logger.debug('Received None answer')
        return ''
    answer = re.sub('\\\\text{[^}]+(?:inches|feet|meters|cm|m|kg|ft|in|lb|oz|ml|L|per|second|minute|hour)[^}]*}', '', answer)
    answer = re.sub('(?<!\\\\)\\s+', '', answer)
    logger.debug(f'After initial whitespace removal: {repr(answer)}')
    ordered_pair_match = re.match('^(?:\\\\left)?\\((.*?)(?:\\\\right)?\\)$', answer)
    if ordered_pair_match:
        content = ordered_pair_match.group(1)
        parts = content.split(',')
        normalized_parts = []
        for part in parts:
            part = re.sub('\\\\?\\s+', '', part)
            norm_part = normalize_answer(part)
            if norm_part is None:
                return None
            normalized_parts.append(norm_part)
        return f'({','.join(normalized_parts)})'
    answer = ''.join(answer.split())
    logger.debug(f'After whitespace removal: {repr(answer)}')
    if not answer:
        logger.debug('Answer became empty after whitespace removal')
        return None
    pm_match = re.match('^(.*?)(?:\\\\pm|-)(.*?)$', answer)
    if pm_match:
        left, right = pm_match.groups()
        norm_left = normalize_answer(left) if left else ''
        norm_right = normalize_answer(right) if right else ''
        if norm_left or norm_right:
            result = f'{norm_left}\\pm{norm_right}'
            logger.debug(f'Matched as plus-minus expression: {repr(result)}')
            return result
    trig_match = re.match('^\\\\(?:sin|cos|tan|cot|sec|csc)\\s*([a-zA-Z])$', answer)
    if trig_match:
        variable = trig_match.group(1)
        func_name = re.match('^\\\\(.*?)(?:\\s|$)', answer).group(1)
        result = f'\\{func_name}{variable}'
        logger.debug(f'Matched as trigonometric function: {repr(result)}')
        return result
    text_match = re.match('^(?:\\\\text{)?([A-Za-z]+)(?:})?$', answer)
    if text_match:
        result = text_match.group(1).lower()
        logger.debug(f'Matched as text answer: {repr(result)}')
        return result
    if (answer.startswith('\\left[') or answer.startswith('\\left(') or answer.startswith('[') or answer.startswith('(')) and (answer.endswith('\\right]') or answer.endswith('\\right)') or answer.endswith(']') or answer.endswith(')')):
        result = normalize_interval(answer)
        if result:
            logger.debug(f'Matched as interval: {repr(result)}')
            return result
    if answer.startswith('\\begin{pmatrix}') and answer.endswith('\\end{pmatrix}'):
        result = normalize_matrix(answer)
        if result:
            logger.debug(f'Matched as matrix: {repr(result)}')
            return result
    answer = answer.replace('\\dfrac', '\\frac')
    if '\\frac' in answer or '\\dfrac' in answer or '/' in answer:
        result = normalize_fraction(answer)
        if result:
            logger.debug(f'Matched as fraction: {repr(result)}')
            return result
    neg_sqrt_match = re.match('^-\\\\sqrt\\{?(\\d+)\\}?$', answer)
    if neg_sqrt_match:
        num = neg_sqrt_match.group(1)
        result = f'-\\sqrt{{{num}}}'
        logger.debug(f'Matched as negative square root: {repr(result)}')
        return result
    logger.debug('Checking for square root pattern...')
    sqrt_match = re.match('^(\\d*)?\\\\sqrt\\{?(\\d+)\\}?$', answer)
    if sqrt_match:
        coeff, num = sqrt_match.groups()
        coeff = coeff if coeff else '1'
        if coeff == '1':
            result = f'\\sqrt{{{num}}}'
        else:
            result = f'{coeff}\\sqrt{{{num}}}'
        logger.debug(f'Matched as pure square root: {repr(result)}')
        return result
    sqrt_with_coeff_match = re.match('^(\\d+)\\\\sqrt\\{?(\\d+)\\}?$', answer)
    if sqrt_with_coeff_match:
        coeff, num = sqrt_with_coeff_match.groups()
        result = f'{coeff}\\sqrt{{{num}}}'
        logger.debug(f'Matched as coefficient with square root: {repr(result)}')
        return result
    base_match = re.match('^(\\d+)(?:_\\{?(\\d+)\\}?|_(\\d+))$', answer)
    if base_match:
        number, base1, base2 = base_match.groups()
        base = base1 if base1 else base2
        result = f'{number}_{base}'
        logger.debug(f'Matched as base number: {repr(result)}')
        return result
    percent_match = re.match('^(\\d+(?:\\.\\d*)?)\\s*\\\\?%$', answer)
    if percent_match:
        number = percent_match.group(1)
        result = normalize_number(number)
        logger.debug(f'Matched as percentage: {repr(result)}')
        return result
    unit_match = re.match('^(\\d+(?:\\.\\d*)?)\\s*(?:(?:\\\\[,\\s])|,)?\\s*(?:\\\\\\\\)?(?:\\\\text{(\\w+)}|\\\\?(?:cm|m|kg|ft|in|lb|oz|ml|L))$', answer)
    if unit_match:
        number = unit_match.group(1)
        result = normalize_number(number)
        logger.debug(f'Matched as number with unit: {repr(result)}')
        return result
    currency_match = re.match('^\\\\?\\$?([\\d,]+\\.?\\d*)$', answer)
    if currency_match:
        result = normalize_number(currency_match.group(1))
        logger.debug(f'Matched as currency: {repr(result)}')
        return result
    if re.match('^-?[\\d,]+$', answer):
        result = normalize_number(answer)
        logger.debug(f'Matched as number: {repr(result)}')
        return result
    unit_match = re.match('^(-?[\\d,]+(?:\\.\\d*)?)\\s*(?:\\\\(?:mbox|text|hbox|displaystyle)\\{[^}]+\\})?(?:\\^?\\d)?$', answer)
    if unit_match:
        result = normalize_number(unit_match.group(1))
        logger.debug(f'Matched as number with units: {repr(result)}')
        return result
    mc_match = re.match('^\\\\text{\\(?([A-Za-z])\\)?}$|^\\(?([A-Za-z])\\)?$', answer)
    if mc_match:
        result = (mc_match.group(1) or mc_match.group(2)).lower()
        logger.debug(f'Matched as multiple choice: {repr(result)}')
        return result
    degree_match = re.match('^(-?[\\d,]+(?:\\.\\d*)?)\\s*(?:(?:\\^?\\\\circ)|(?:{\\\\circ})|(?:°))?$', answer)
    if degree_match:
        result = normalize_number(degree_match.group(1))
        logger.debug(f'Matched as degrees: {repr(result)}')
        return result
    answer = re.sub('\\\\text{([^{}]+)}', '\\1', answer)
    logger.debug(f'After \\text removal: {repr(answer)}')
    try:
        result = normalize_algebraic_expression(answer)
        logger.debug(f'Normalized as algebraic expression: {repr(result)}')
        return result
    except:
        logger.debug('Failed to normalize as algebraic expression')
        pass
    answer = answer.replace('\\left', '').replace('\\right', '')
    answer = answer.replace('\\left', '').replace('\\right', '')
    answer = answer.replace('\\(', '(').replace('\\)', ')')
    answer = answer.replace('\\[', '[').replace('\\]', ']')
    answer = answer.replace('\\{', '{').replace('\\}', '}')
    answer = re.sub('\\\\sqrt\\{?(\\d+)\\}?', '\\\\sqrt{\\1}', answer)
    answer = re.sub('\\\\sqrt{([^{}]+)}', '\\\\sqrt\\1', answer)
    if re.match('^\\d+\\\\%$', answer) or re.match('^\\d+$', answer):
        answer = re.sub('\\\\%$', '', answer)
    answer = re.sub('\\\\text{([^{}]+)}', '\\1', answer)
    while len(answer) >= 2 and answer[0] == '{' and (answer[-1] == '}'):
        if '\\frac' in answer:
            break
        answer = answer[1:-1]
    result = answer.lower()
    logger.debug(f'Final normalized result: {repr(result)}')
    return result if result else None

def normalize_fraction(fraction_str: str) -> str:
    """Helper function to normalize fractions."""
    logger.debug(f'Normalizing fraction: {repr(fraction_str)}')
    try:
        fraction_str = fraction_str.replace('\\dfrac', '\\frac')
        fraction_str = ''.join(fraction_str.split())
        fraction_str = re.sub('\\s*\\\\text{[^}]+}', '', fraction_str)
        mixed_brace = re.match('^\\\\frac(\\d+)\\{(\\d+)\\}$', fraction_str)
        if mixed_brace:
            num, den = mixed_brace.groups()
            return f'\\frac{{{num}}}{{{den}}}'
        no_braces = re.match('^\\\\frac(\\d+)(\\d+)$', fraction_str)
        if no_braces:
            num, den = no_braces.groups()
            return f'\\frac{{{num}}}{{{den}}}'
        if '/' in fraction_str and (not any((c in fraction_str for c in '\\{}'))):
            num, den = fraction_str.split('/')
            return f'\\frac{{{num.strip()}}}{{{den.strip()}}}'
        standard = re.match('^\\\\frac\\{([^{}]+)\\}\\{([^{}]+)\\}$', fraction_str)
        if standard:
            num, den = standard.groups()
            return f'\\frac{{{num}}}{{{den}}}'
    except Exception as e:
        logger.debug(f'Failed to normalize fraction: {str(e)}')
        logger.debug(f'Original fraction string: {repr(fraction_str)}')
    return fraction_str

def compare_answers(correct_answer: str, predicted_answer: Optional[str]) -> bool:
    """Compare the correct answer with the predicted answer."""
    logger.debug(f'Comparing answers - Correct: {repr(correct_answer)}, Predicted: {repr(predicted_answer)}')
    if predicted_answer is None:
        logger.debug('Predicted answer is None')
        return False
    if numerically_equal(correct_answer, predicted_answer):
        return True
    normalized_correct = normalize_answer(correct_answer)
    normalized_predicted = normalize_answer(predicted_answer)
    logger.debug(f'Normalized answers - Correct: {repr(normalized_correct)}, Predicted: {repr(normalized_predicted)}')
    if not normalized_correct or not normalized_predicted:
        logger.debug('One or both normalized answers are None or empty')
        return False
    if normalized_correct == '' and normalized_predicted == '':
        logger.debug('Both answers normalized to empty strings')
        return False
    if ('\\left[' in normalized_correct or '\\left(' in normalized_correct) and ('\\left[' in normalized_predicted or '\\left(' in normalized_predicted):
        result = normalized_correct == normalized_predicted
        logger.debug(f'Interval comparison result: {result}')
        return result
    result = normalized_correct == normalized_predicted
    logger.debug(f'Comparison result: {result}')
    return result

def numerically_equal(str1: str, str2: str) -> bool:
    """Compare if two numeric strings represent the same value."""
    try:
        return abs(float(str1) - float(str2)) < 1e-10
    except:
        return False

