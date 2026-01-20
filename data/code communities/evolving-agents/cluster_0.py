# Cluster 0

def extract_final_json_answer(response_text: str) -> Optional[Dict[str, Any]]:
    """Extracts the JSON object that an agent often provides as its 'Final Answer:'."""
    json_block_match = re.search('```json\\s*(\\{[\\s\\S]*?\\})\\s*```', response_text, re.MULTILINE | re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1).strip())
        except json.JSONDecodeError:
            logger.warning('Found JSON block but failed to parse in extract_final_json_answer.')
    potential_json_match = re.search('(?i)(?:Final Answer:|Output:|Here is the JSON output:|Response:)\\s*(\\{[\\s\\S]*\\})(?:\\s*```)?\\s*$', response_text, re.MULTILINE | re.DOTALL)
    if potential_json_match:
        try:
            return json.loads(potential_json_match.group(1).strip())
        except json.JSONDecodeError:
            logger.warning('Found potential final JSON but failed to parse in extract_final_json_answer.')
    try:
        if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
            return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    logger.warning(f'Could not extract final JSON answer from response: {response_text[:200]}...')
    return None

def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    if not response_text:
        return None
    patterns = ['```json\\s*(\\{[\\s\\S]*?\\})\\s*```', '```\\s*(\\{[\\s\\S]*?\\})\\s*```']
    for pattern in patterns:
        match = re.search(pattern, response_text, re.MULTILINE | re.DOTALL)
        if match:
            try:
                potential_json_str = match.group(1).strip()
                if potential_json_str.startswith('{') and potential_json_str.endswith('}') and (potential_json_str.count('{') >= potential_json_str.count('}')):
                    return json.loads(potential_json_str)
            except json.JSONDecodeError:
                logger.debug(f"Pattern '{pattern}' matched but failed to parse as JSON: '{match.group(1)[:100]}...'")
                continue
    json_objects_found = []
    for match in re.finditer('(\\{[\\s\\S]*?\\})(?=\\s*\\{|\\Z)', response_text, re.DOTALL):
        potential_json_str = match.group(1).strip()
        try:
            if potential_json_str.startswith('{') and potential_json_str.endswith('}') and (potential_json_str.count('{') >= potential_json_str.count('}')):
                if ':' in potential_json_str and ('"' in potential_json_str or "'" in potential_json_str):
                    parsed = json.loads(potential_json_str)
                    if isinstance(parsed, dict) and len(parsed.keys()) > 1:
                        json_objects_found.append(parsed)
        except json.JSONDecodeError:
            logger.debug(f"Loose JSON match failed to parse: '{potential_json_str[:100]}...'")
    if json_objects_found:
        return max(json_objects_found, key=lambda x: len(json.dumps(x)))
    logger.warning('Could not extract JSON using pattern matching from SystemAgent response.')
    return None

def get_nested_value(data_dict: Dict[str, Any], keys_list_to_check: List[str], default_val: Any='N/A') -> Any:
    """
    Retrieves a value from a dictionary using a list of possible keys.
    Performs case-insensitive and space-insensitive matching for keys, also removing '#'.
    """
    if not isinstance(data_dict, dict):
        return default_val
    normalized_data_keys_map = {key.lower().replace(' ', '').replace('#', ''): key for key in data_dict.keys() if isinstance(key, str)}
    for key_to_try in keys_list_to_check:
        if key_to_try in data_dict:
            return data_dict[key_to_try]
        normalized_key_to_try = key_to_try.lower().replace(' ', '').replace('#', '')
        if normalized_key_to_try in normalized_data_keys_map:
            original_key_in_dict = normalized_data_keys_map[normalized_key_to_try]
            return data_dict[original_key_in_dict]
    return default_val

