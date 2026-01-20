# Cluster 4

def main():
    logging.info('STARTING extracts.py')
    transcript, subtitles = get_whisper_output()
    if transcript is None or subtitles is None:
        logging.error('Failed to get whisper output')
        return None
    response = call_openai_api(transcript)
    if response and 'clips' in response:
        output_dir = Path('crew_output')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'api_response.json'
        save_response_to_file(response, output_path)
        extracts = [clip['text'] for clip in response['clips']]
        return extracts
    else:
        logging.error('Failed to get a valid response from OpenAI API')
        if response:
            logging.error(f'Unexpected response structure: {response}')
        return None

def get_whisper_output():
    whisper_output_dir = Path('whisper_output')
    if not whisper_output_dir.exists():
        logging.error(f'Directory not found: {whisper_output_dir}')
        return (None, None)
    srt_files = list(whisper_output_dir.glob('*.srt'))
    txt_files = list(whisper_output_dir.glob('*.txt'))
    if not srt_files or not txt_files:
        logging.warning('No .srt or .txt files found in the whisper_output directory.')
        return (None, None)
    with open(txt_files[0], 'r') as file:
        transcript = file.read()
    with open(srt_files[0], 'r') as file:
        subtitles = file.read()
    return (transcript, subtitles)

def call_openai_api(transcript):
    logging.info('STARTING call_openai_api')
    prompt = dedent(f'\n        You will be given a complete transcript from a video. Your task is to identify four 1-minute long clips from this video that have the highest potential to become popular on social media. \n        \n        Follow these steps to complete the task:\n        \n        1. Carefully read through the entire transcript, looking for the most powerful, emotionally impactful, surprising, thought-provoking, or otherwise memorable moments. Give priority to answers and speculations rather than questions.\n        \n        2. For each standout moment you identify, extract a 1-minute segment of text from the transcript, centered around that moment. Ensure each segment is approximately 1 minute long when spoken (about 125 words or 10 spoken sentences).\n        \n        3. From these segments, choose the top four that you believe have the highest potential to go viral on social media.\n        \n        4. Rank these four clips from most to least viral potential based on your assessment.\n        \n        5. Determine the word count for each of the four selected clips.\n        \n        6. Format your final output as a JSON object containing an ordered list of the selected clips, each with its extracted text. The JSON object should look like this:\n        \n        {{\n            "clips": [\n            {{\n                "rank": 1,\n                "text": "<extracted text for clip 1>",\n                "wordcount": <length of the extracted text in words>\n            }},\n            {{\n                "rank": 2,\n                "text": "<extracted text for clip 2>",\n                "wordcount": <length of the extracted text in words>\n            }},\n            {{\n                "rank": 3,\n                "text": "<extracted text for clip 3>",\n                "wordcount": <length of the extracted text in words>\n            }},\n            {{\n                "rank": 4,\n                "text": "<extracted text for clip 4>",\n                "wordcount": <length of the extracted text in words>\n            }}\n            ]\n        }}\n        \n        Here is the transcript:\n        \n        <transcript>\n        {transcript}\n        </transcript>\n        \n        Important reminders:\n        - Always return EXACTLY THREE clips.\n        - DO NOT OMIT any text\n        - Return nothing else but the raw content of the JSON object itself - no comments, no extra text. Just the JSON.\n        - Ensure that each clip is approximately 1 minute long when spoken (about 125 words or 10 spoken sentences).\n        - Focus on selecting clips that are powerful, emotionally impactful, surprising, thought-provoking, or otherwise memorable.\n        - Prioritize answers and speculations over questions when selecting clips.\n    ')
    try:
        response = client.chat.completions.create(model='gpt-4o-2024-08-06', messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt}], temperature=0.8, max_tokens=4095, top_p=1, frequency_penalty=0, presence_penalty=0, response_format={'type': 'json_schema', 'json_schema': {'name': 'clips_response', 'strict': True, 'schema': {'type': 'object', 'properties': {'clips': {'type': 'array', 'items': {'type': 'object', 'properties': {'rank': {'type': 'integer'}, 'text': {'type': 'string'}, 'wordcount': {'type': 'integer'}}, 'required': ['rank', 'text', 'wordcount'], 'additionalProperties': False}}}, 'required': ['clips'], 'additionalProperties': False}}})
        if not response or not response.choices:
            logging.error('No response or choices from OpenAI API')
            return None
        response_text = response.choices[0].message.content
        logging.info(f'Raw API response: {response_text}')
        try:
            response_data = json.loads(response_text)
            if len(response_data.get('clips', [])) != 3:
                logging.error('The response does not contain exactly four clips. Generating filler content.')
                while len(response_data['clips']) < 3:
                    response_data['clips'].append({'rank': len(response_data['clips']) + 1, 'text': 'This is filler content to ensure there are exactly four clips.'})
            return response_data
        except json.JSONDecodeError as e:
            logging.error(f'JSON Decode Error: {str(e)}')
            logging.error(f'Problematic JSON string: {response_text}')
            logging.error(f'First 500 characters of response: {response_text[:500]}')
            return None
    except Exception as e:
        logging.error(f'Error calling OpenAI API: {str(e)}')
        logging.error(traceback.format_exc())
        return None

def save_response_to_file(response, output_path):
    try:
        with open(output_path, 'w') as f:
            json.dump(response, f, indent=4)
        logging.info(f'Response saved to {output_path}')
    except Exception as e:
        logging.error(f'Error saving response to file: {e}')

