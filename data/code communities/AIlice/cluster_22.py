# Cluster 22

@app.route('/chat', methods=['POST'])
def chat():
    with lock:
        message = request.get_json().get('message', '')
        return Response(generate_response(message), mimetype='text/event-stream')

def generate_response(message):
    try:
        context[currentSession]['processor'].SetGas(amount=100000000.0)
        threadLLM = threading.Thread(target=context[currentSession]['processor'], args=(message,))
        threadLLM.start()
        depth = -1
        braketMap = {'<': 1, '>': -1}
        while True:
            channel, txt, action = context[currentSession]['logger'].queue.get()
            depth += braketMap.get(channel, 0)
            if -1 == depth and '>' == channel:
                threadLLM.join()
                with open(os.path.join(config.chatHistoryPath, currentSession, 'ailice_history.json'), 'w') as f:
                    json.dump(context[currentSession]['processor'].ToJson(), f, indent=2)
                return
            elif channel in ['<', '>']:
                continue
            msg = json.dumps({'message': txt, 'role': channel, 'action': action, 'msgType': 'internal' if depth > 0 else 'user-ailice'})
            yield f'data: {msg}\n\n'
    except Exception as e:
        app.logger.error(f'Error in generate_response: {e} {traceback.print_tb(e.__traceback__)}')
        yield f'data: Error occurred: {e}\n\n'

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    with lock:
        if 'audio' not in request.files:
            return (jsonify({'error': 'No audio file provided'}), 400)
        audio = request.files['audio']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio.filename))
        audio.save(filepath)
        if config.speechOn:
            audio_data, sample_rate = librosa.load(filepath)
            message = speech.Speech2Text(audio_data.tolist(), sample_rate)
        else:
            message = f'![audio]({str(filepath)})'
        return Response(generate_response(f'{message}'), mimetype='text/event-stream')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    with lock:
        if 'image' not in request.files:
            return (jsonify({'error': 'No image file provided'}), 400)
        image = request.files['image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
        image.save(filepath)
        return Response(generate_response(f'![image]({str(filepath)})'), mimetype='text/event-stream')

