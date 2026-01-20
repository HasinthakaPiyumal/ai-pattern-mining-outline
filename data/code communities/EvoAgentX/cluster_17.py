# Cluster 17

def validate_parameter(model: str, param: str, value: any, operation: str='generation') -> Tuple[bool, str]:
    config = get_model_config(model, operation)
    if not config:
        return (False, f'Unsupported model: {model}')
    if param not in config.get('supported_params', []):
        return (False, f"Parameter '{param}' is not supported by {model}")
    if param == 'size' and value not in config.get('size_options', []):
        return (False, f"Invalid size '{value}' for {model}. Supported: {config['size_options']}")
    if param == 'quality' and value not in config.get('quality_options', []):
        return (False, f"Invalid quality '{value}' for {model}. Supported: {config['quality_options']}")
    if param == 'style' and value not in config.get('style_options', []):
        return (False, f"Invalid style '{value}' for {model}. Supported: {config['style_options']}")
    if param == 'background' and value not in config.get('background_options', []):
        return (False, f"Invalid background '{value}' for {model}. Supported: {config['background_options']}")
    if param == 'moderation' and value not in config.get('moderation_options', []):
        return (False, f"Invalid moderation '{value}' for {model}. Supported: {config['moderation_options']}")
    if param == 'input_fidelity' and value not in config.get('input_fidelity_options', []):
        return (False, f"Invalid input_fidelity '{value}' for {model}. Supported: {config['input_fidelity_options']}")
    if param == 'output_format' and value not in config.get('output_format_options', []):
        return (False, f"Invalid output_format '{value}'. Supported: {config['output_format_options']}")
    if param == 'n' and value > config.get('n_max', 10):
        return (False, f'Invalid n {value}. Max: {config['n_max']}')
    if param == 'output_compression' and (value < 0 or value > 100):
        return (False, 'output_compression must be between 0 and 100')
    if param == 'partial_images' and (value < 0 or value > 3):
        return (False, 'partial_images must be between 0 and 3')
    return (True, '')

def get_model_config(model: str, operation: str='generation') -> Dict:
    if operation == 'editing':
        return OPENAI_EDITING_MODEL_CONFIG.get(model, {})
    return OPENAI_MODEL_CONFIG.get(model, {})

def validate_parameters(model: str, params: Dict, operation: str='generation') -> Dict:
    validated_params = {}
    warnings = []
    errors = []
    for param, value in params.items():
        if value is None:
            continue
        ok, msg = validate_parameter(model, param, value, operation)
        if ok:
            validated_params[param] = value
        elif 'not supported' in msg:
            warnings.append(msg)
        else:
            errors.append(msg)
    return {'validated_params': validated_params, 'warnings': warnings, 'errors': errors}

class OpenAIImageGenerationTool(Tool):
    name: str = 'openai_image_generation'
    description: str = 'OpenAI image generation supporting dall-e-2, dall-e-3, gpt-image-1 (with validation).'
    inputs: Dict[str, Dict[str, str]] = {'prompt': {'type': 'string', 'description': 'Prompt text. Required.'}, 'image_name': {'type': 'string', 'description': 'Optional save name.'}, 'model': {'type': 'string', 'description': 'dall-e-2 | dall-e-3 | gpt-image-1'}, 'size': {'type': 'string', 'description': 'Model-specific size.'}, 'quality': {'type': 'string', 'description': 'quality for gpt-image-1/dall-e-3'}, 'n': {'type': 'integer', 'description': '1-10 (1 for dalle-3)'}, 'background': {'type': 'string', 'description': 'gpt-image-1 only'}, 'moderation': {'type': 'string', 'description': 'gpt-image-1 only'}, 'output_compression': {'type': 'integer', 'description': 'gpt-image-1 jpeg/webp'}, 'output_format': {'type': 'string', 'description': 'gpt-image-1 png/jpeg/webp'}, 'partial_images': {'type': 'integer', 'description': 'gpt-image-1 streaming partials'}, 'response_format': {'type': 'string', 'description': 'url | b64_json for dalle-2/3'}, 'stream': {'type': 'boolean', 'description': 'gpt-image-1 streaming'}, 'style': {'type': 'string', 'description': 'dall-e-3 vivid|natural'}}
    required: Optional[List[str]] = ['prompt']

    def __init__(self, api_key: str, organization_id: str=None, model: str='dall-e-3', save_path: str='./generated_images', storage_handler: Optional[FileStorageHandler]=None):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)

    def __call__(self, prompt: str, image_name: str=None, model: str=None, size: str=None, quality: str=None, n: int=None, background: str=None, moderation: str=None, output_compression: int=None, output_format: str=None, partial_images: int=None, response_format: str=None, stream: bool=None, style: str=None):
        try:
            client = create_openai_client(self.api_key, self.organization_id)
            actual_model = model if model else self.model
            params_to_validate = build_validation_params(model=actual_model, prompt=prompt, size=size, quality=quality, n=n, background=background, moderation=moderation, output_compression=output_compression, output_format=output_format, partial_images=partial_images, response_format=response_format, stream=stream, style=style)
            validation_result = validate_parameters(actual_model, params_to_validate, 'generation')
            error = handle_validation_result(validation_result)
            if error:
                return error
            api_params = validation_result['validated_params'].copy()
            api_params.pop('image_name', None)
            response = client.images.generate(**api_params)
            import base64
            results = []
            for i, image_data in enumerate(response.data):
                try:
                    if hasattr(image_data, 'b64_json') and image_data.b64_json:
                        image_bytes = base64.b64decode(image_data.b64_json)
                    elif hasattr(image_data, 'url') and image_data.url:
                        import requests
                        r = requests.get(image_data.url)
                        r.raise_for_status()
                        image_bytes = r.content
                    else:
                        raise Exception('No valid image data in response')
                    filename = self._get_unique_filename(image_name, i)
                    result = self.storage_handler.save(filename, image_bytes)
                    if result['success']:
                        results.append(filename)
                    else:
                        results.append(f'Error saving image {i + 1}: {result.get('error', 'Unknown error')}')
                except Exception as e:
                    results.append(f'Error saving image {i + 1}: {e}')
            return {'results': results, 'count': len(results)}
        except Exception as e:
            return {'error': f'Image generation failed: {e}'}

    def _get_unique_filename(self, image_name: str, index: int) -> str:
        """Generate a unique filename for the image"""
        import time
        if image_name:
            base = image_name.rsplit('.', 1)[0]
            filename = f'{base}_{index + 1}.png'
        else:
            ts = int(time.time())
            filename = f'generated_{ts}_{index + 1}.png'
        counter = 1
        while self.storage_handler.exists(filename):
            if image_name:
                base = image_name.rsplit('.', 1)[0]
                filename = f'{base}_{index + 1}_{counter}.png'
            else:
                filename = f'generated_{ts}_{index + 1}_{counter}.png'
            counter += 1
        return filename

def create_openai_client(api_key: str, organization_id: str | None=None):
    from openai import OpenAI
    return OpenAI(api_key=api_key, organization=organization_id)

def build_validation_params(**kwargs) -> Dict:
    return {k: v for k, v in kwargs.items() if v is not None}

def handle_validation_result(validation_result: Dict) -> Dict | None:
    if validation_result['errors']:
        return {'error': f'Parameter validation failed: {'; '.join(validation_result['errors'])}'}
    if validation_result['warnings']:
        print(f'⚠️ Parameter warnings: {'; '.join(validation_result['warnings'])}')
        print('📝 Note: Continue with supported parameters only')
    return None

class OpenAIImageEditTool(Tool):
    name: str = 'openai_image_edit'
    description: str = 'Edit images using OpenAI gpt-image-1 (direct, minimal validation).'
    inputs: Dict[str, Dict[str, str]] = {'prompt': {'type': 'string', 'description': 'Edit instruction. Required.'}, 'images': {'type': 'array', 'description': 'Image path(s) png/webp/jpg <50MB. Required. Single string accepted and normalized to array.'}, 'mask_path': {'type': 'string', 'description': 'Optional PNG mask path (same size as first image).'}, 'size': {'type': 'string', 'description': '1024x1024 | 1536x1024 | 1024x1536 | auto'}, 'n': {'type': 'integer', 'description': '1-10'}, 'background': {'type': 'string', 'description': 'transparent | opaque | auto'}, 'input_fidelity': {'type': 'string', 'description': 'high | low'}, 'output_compression': {'type': 'integer', 'description': '0-100 for jpeg/webp'}, 'output_format': {'type': 'string', 'description': 'png | jpeg | webp (default png)'}, 'partial_images': {'type': 'integer', 'description': '0-3 partial streaming'}, 'quality': {'type': 'string', 'description': 'auto | high | medium | low'}, 'stream': {'type': 'boolean', 'description': 'streaming mode'}, 'image_name': {'type': 'string', 'description': 'Optional output base name'}}
    required: Optional[List[str]] = ['prompt', 'images']

    def __init__(self, api_key: str, organization_id: str=None, save_path: str='./edited_images', storage_handler: Optional[FileStorageHandler]=None):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)

    def __call__(self, prompt: str, images: list, mask_path: str=None, size: str=None, n: int=None, background: str=None, input_fidelity: str=None, output_compression: int=None, output_format: str=None, partial_images: int=None, quality: str=None, stream: bool=None, image_name: str=None):
        try:
            client = create_openai_client(self.api_key, self.organization_id)
            if isinstance(images, str):
                image_paths = [images]
            else:
                image_paths = list(images)
            opened_images = []
            temp_paths = []
            mask_fh = None
            try:
                for p in image_paths:
                    use_path, tmp = self._ensure_image_edit_compatible(p)
                    if tmp:
                        temp_paths.append(tmp)
                    opened_images.append(open(use_path, 'rb'))
                api_kwargs = {'model': 'gpt-image-1', 'prompt': prompt, 'image': opened_images if len(opened_images) > 1 else opened_images[0]}
                if size is not None:
                    api_kwargs['size'] = size
                if n is not None:
                    api_kwargs['n'] = n
                if background is not None:
                    api_kwargs['background'] = background
                if input_fidelity is not None:
                    api_kwargs['input_fidelity'] = input_fidelity
                if output_compression is not None:
                    api_kwargs['output_compression'] = output_compression
                if output_format is not None:
                    api_kwargs['output_format'] = output_format
                if partial_images is not None:
                    api_kwargs['partial_images'] = partial_images
                if quality is not None:
                    api_kwargs['quality'] = quality
                if stream is not None:
                    api_kwargs['stream'] = stream
                if mask_path:
                    mask_fh = open(mask_path, 'rb')
                    api_kwargs['mask'] = mask_fh
                response = client.images.edit(**api_kwargs)
            finally:
                for fh in opened_images:
                    try:
                        fh.close()
                    except Exception:
                        pass
                if mask_fh:
                    try:
                        mask_fh.close()
                    except Exception:
                        pass
                import os
                for tp in temp_paths:
                    try:
                        if tp and os.path.exists(tp):
                            os.remove(tp)
                    except Exception:
                        pass
            import base64
            import time
            results = []
            for i, img in enumerate(response.data):
                try:
                    img_bytes = base64.b64decode(img.b64_json)
                    ts = int(time.time())
                    if image_name:
                        filename = f'{image_name.rsplit('.', 1)[0]}_{i + 1}.png'
                    else:
                        filename = f'image_edit_{ts}_{i + 1}.png'
                    result = self.storage_handler.save(filename, img_bytes)
                    if result['success']:
                        translated_path = self.storage_handler.translate_in(filename)
                        results.append(translated_path)
                    else:
                        results.append(f'Error saving image {i + 1}: {result.get('error', 'Unknown error')}')
                except Exception as e:
                    results.append(f'Error saving image {i + 1}: {e}')
            return {'results': results, 'count': len(results)}
        except Exception as e:
            return {'error': f'gpt-image-1 editing failed: {e}'}

    def _ensure_image_edit_compatible(self, image_path: str) -> tuple[str, str | None]:
        """
        Ensure the image matches OpenAI edit requirements using storage handler.
        If not, convert to RGBA and save to a temporary path. Return (usable_path, temp_path).
        Caller may delete temp_path after the request completes.
        """
        try:
            from PIL import Image
            from io import BytesIO
            import os
            result = self.storage_handler.read(image_path)
            if not result['success']:
                raise FileNotFoundError(f'Could not read image {image_path}: {result.get('error', 'Unknown error')}')
            if isinstance(result['content'], bytes):
                content = result['content']
            else:
                content = str(result['content']).encode('utf-8')
            with Image.open(BytesIO(content)) as img:
                if img.mode in ('RGBA', 'LA', 'L'):
                    translated_path = self.storage_handler.translate_in(image_path)
                    return (translated_path, None)
                rgba_img = img.convert('RGBA')
                temp_filename = f'temp_rgba_{hash(image_path) % 10000}.png'
                buffer = BytesIO()
                rgba_img.save(buffer, format='PNG')
                temp_content = buffer.getvalue()
                result = self.storage_handler.save(temp_filename, temp_content)
                if result['success']:
                    temp_path = self.storage_handler.translate_in(temp_filename)
                    return (temp_path, temp_path)
                else:
                    temp_path = os.path.join('workplace', 'images', 'temp_rgba_image.png')
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                    rgba_img.save(temp_path)
                    return (temp_path, temp_path)
        except Exception:
            translated_path = self.storage_handler.translate_in(image_path)
            return (translated_path, None)

class OpenAIImageAnalysisTool(Tool):
    name: str = 'openai_image_analysis'
    description: str = 'Simple image analysis via OpenAI Responses API (input_text + input_image).'
    inputs: Dict[str, Dict[str, str]] = {'prompt': {'type': 'string', 'description': 'User question/instruction. Required.'}, 'image_url': {'type': 'string', 'description': 'HTTP(S) image URL. Optional if image_path provided.'}, 'image_path': {'type': 'string', 'description': 'Local image path; converted to data URL internally.'}, 'model': {'type': 'string', 'description': 'OpenAI model for responses.create (e.g., gpt-4o-mini, gpt-4.1, gpt-5). Optional.'}}
    required: Optional[List[str]] = ['prompt']

    def __init__(self, api_key: str, organization_id: str=None, model: str='gpt-4o-mini', storage_handler: Optional[FileStorageHandler]=None):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.model = model
        self.storage_handler = storage_handler or LocalStorageHandler()

    def __call__(self, prompt: str, image_url: str=None, image_path: str=None, model: str=None):
        try:
            client = create_openai_client(self.api_key, self.organization_id)
            actual_model = model if model else self.model
            final_image_url = image_url
            if not final_image_url and image_path:
                import base64
                import mimetypes
                mime, _ = mimetypes.guess_type(image_path)
                mime = mime or 'image/png'
                try:
                    system_path = self.storage_handler.translate_in(image_path)
                    content = self.storage_handler._read_raw(system_path)
                except Exception as e:
                    return {'error': f'Could not read image {image_path}: {str(e)}'}
                b64 = base64.b64encode(content).decode('utf-8')
                final_image_url = f'data:{mime};base64,{b64}'
            response = client.responses.create(model=actual_model, input=[{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}, {'type': 'input_image', 'image_url': final_image_url}]}])
            text = getattr(response, 'output_text', None)
            if text is None:
                try:
                    choices = getattr(response, 'output', None) or getattr(response, 'choices', None)
                    if choices and isinstance(choices, list):
                        first = choices[0]
                        text = getattr(first, 'message', {}).get('content', '') if isinstance(first, dict) else ''
                except Exception:
                    text = ''
            return {'content': text or ''}
        except Exception as e:
            return {'error': f'OpenAI image analysis failed: {e}'}

