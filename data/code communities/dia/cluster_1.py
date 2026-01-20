# Cluster 1

def run_inference(text_input: str, audio_prompt_text_input: str, audio_prompt_input: Optional[Tuple[int, np.ndarray]], max_new_tokens: int, cfg_scale: float, temperature: float, top_p: float, cfg_filter_top_k: int, speed_factor: float, seed: Optional[int]=None):
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Uses temporary files for text and audio prompt compatibility with inference.generate.
    """
    global model, device
    console_output_buffer = io.StringIO()
    with contextlib.redirect_stdout(console_output_buffer):
        if audio_prompt_input and audio_prompt_text_input and (not audio_prompt_text_input.isspace()):
            text_input = audio_prompt_text_input + '\n' + text_input
            text_input = text_input.strip()
        if audio_prompt_input and (not audio_prompt_text_input or audio_prompt_text_input.isspace()):
            raise gr.Error('Audio Prompt Text input cannot be empty.')
        if not text_input or text_input.isspace():
            raise gr.Error('Text input cannot be empty.')
        temp_txt_file_path = None
        temp_audio_prompt_path = None
        output_audio = (44100, np.zeros(1, dtype=np.float32))
        try:
            prompt_path_for_generate = None
            if audio_prompt_input is not None:
                sr, audio_data = audio_prompt_input
                if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:
                    gr.Warning('Audio prompt seems empty or silent, ignoring prompt.')
                else:
                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as f_audio:
                        temp_audio_prompt_path = f_audio.name
                        if np.issubdtype(audio_data.dtype, np.integer):
                            max_val = np.iinfo(audio_data.dtype).max
                            audio_data = audio_data.astype(np.float32) / max_val
                        elif not np.issubdtype(audio_data.dtype, np.floating):
                            gr.Warning(f'Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion.')
                            try:
                                audio_data = audio_data.astype(np.float32)
                            except Exception as conv_e:
                                raise gr.Error(f'Failed to convert audio prompt to float32: {conv_e}')
                        if audio_data.ndim > 1:
                            if audio_data.shape[0] == 2:
                                audio_data = np.mean(audio_data, axis=0)
                            elif audio_data.shape[1] == 2:
                                audio_data = np.mean(audio_data, axis=1)
                            else:
                                gr.Warning(f'Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis.')
                                audio_data = audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
                            audio_data = np.ascontiguousarray(audio_data)
                        try:
                            sf.write(temp_audio_prompt_path, audio_data, sr, subtype='FLOAT')
                            prompt_path_for_generate = temp_audio_prompt_path
                            print(f'Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})')
                        except Exception as write_e:
                            print(f'Error writing temporary audio file: {write_e}')
                            raise gr.Error(f'Failed to save audio prompt: {write_e}')
            if seed is None or seed < 0:
                seed = random.randint(0, 2 ** 32 - 1)
                print(f'\nNo seed provided, generated random seed: {seed}\n')
            else:
                print(f'\nUsing user-selected seed: {seed}\n')
            set_seed(seed)
            print(f'Generating speech: \n"{text_input}"\n')
            start_time = time.time()
            with torch.inference_mode():
                output_audio_np = model.generate(text_input, max_tokens=max_new_tokens, cfg_scale=cfg_scale, temperature=temperature, top_p=top_p, cfg_filter_top_k=cfg_filter_top_k, use_torch_compile=False, audio_prompt=prompt_path_for_generate, verbose=True)
            end_time = time.time()
            print(f'Generation finished in {end_time - start_time:.2f} seconds.\n')
            if output_audio_np is not None:
                output_sr = 44100
                original_len = len(output_audio_np)
                speed_factor = max(0.1, min(speed_factor, 5.0))
                target_len = int(original_len / speed_factor)
                if target_len != original_len and target_len > 0:
                    x_original = np.arange(original_len)
                    x_resampled = np.linspace(0, original_len - 1, target_len)
                    resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                    output_audio = (output_sr, resampled_audio_np.astype(np.float32))
                    print(f'Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.')
                else:
                    output_audio = (output_sr, output_audio_np)
                    print(f'Skipping audio speed adjustment (factor: {speed_factor:.2f}).')
                print(f'Audio conversion successful. Final shape: {output_audio[1].shape}, Sample Rate: {output_sr}')
                if output_audio[1].dtype == np.float32 or output_audio[1].dtype == np.float64:
                    audio_for_gradio = np.clip(output_audio[1], -1.0, 1.0)
                    audio_for_gradio = (audio_for_gradio * 32767).astype(np.int16)
                    output_audio = (output_sr, audio_for_gradio)
                    print('Converted audio to int16 for Gradio output.')
            else:
                print('\nGeneration finished, but no valid tokens were produced.')
                gr.Warning('Generation produced no output.')
        except Exception as e:
            print(f'Error during inference: {e}')
            import traceback
            traceback.print_exc()
            raise gr.Error(f'Inference failed: {e}')
        finally:
            if temp_txt_file_path and Path(temp_txt_file_path).exists():
                try:
                    Path(temp_txt_file_path).unlink()
                    print(f'Deleted temporary text file: {temp_txt_file_path}')
                except OSError as e:
                    print(f'Warning: Error deleting temporary text file {temp_txt_file_path}: {e}')
            if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
                try:
                    Path(temp_audio_prompt_path).unlink()
                    print(f'Deleted temporary audio prompt file: {temp_audio_prompt_path}')
                except OSError as e:
                    print(f'Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}')
        console_output = console_output_buffer.getvalue()
    return (output_audio, seed, console_output)

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Generate audio using the Dia model.')
    parser.add_argument('text', type=str, help='Input text for speech generation.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the generated audio file (e.g., output.wav).')
    parser.add_argument('--repo-id', type=str, default='nari-labs/Dia-1.6B-0626', help='Hugging Face repository ID (e.g., nari-labs/Dia-1.6B-0626).')
    parser.add_argument('--local-paths', action='store_true', help='Load model from local config and checkpoint files.')
    parser.add_argument('--config', type=str, help='Path to local config.json file (required if --local-paths is set).')
    parser.add_argument('--checkpoint', type=str, help='Path to local model checkpoint .pth file (required if --local-paths is set).')
    parser.add_argument('--audio-prompt', type=str, default=None, help='Path to an optional audio prompt WAV file for voice cloning.')
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument('--max-tokens', type=int, default=None, help='Maximum number of audio tokens to generate (defaults to config value).')
    gen_group.add_argument('--cfg-scale', type=float, default=3.0, help='Classifier-Free Guidance scale (default: 3.0).')
    gen_group.add_argument('--temperature', type=float, default=1.3, help='Sampling temperature (higher is more random, default: 0.7).')
    gen_group.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling probability (default: 0.95).')
    infra_group = parser.add_argument_group('Infrastructure')
    infra_group.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    infra_group.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run inference on (e.g., 'cuda', 'cpu', default: auto).")
    args = parser.parse_args()
    if args.local_paths:
        if not args.config:
            parser.error('--config is required when --local-paths is set.')
        if not args.checkpoint:
            parser.error('--checkpoint is required when --local-paths is set.')
        if not os.path.exists(args.config):
            parser.error(f'Config file not found: {args.config}')
        if not os.path.exists(args.checkpoint):
            parser.error(f'Checkpoint file not found: {args.checkpoint}')
    if args.seed is not None:
        set_seed(args.seed)
        print(f'Using user-selected seed: {args.seed}')
    device = torch.device(args.device)
    print(f'Using device: {device}')
    print('Loading model...')
    if args.local_paths:
        print(f"Loading from local paths: config='{args.config}', checkpoint='{args.checkpoint}'")
        try:
            model = Dia.from_local(args.config, args.checkpoint, device=device)
        except Exception as e:
            print(f'Error loading local model: {e}')
            exit(1)
    else:
        print(f"Loading from Hugging Face Hub: repo_id='{args.repo_id}'")
        try:
            model = Dia.from_pretrained(args.repo_id, device=device)
        except Exception as e:
            print(f'Error loading model from Hub: {e}')
            exit(1)
    print('Model loaded.')
    print('Generating audio...')
    try:
        sample_rate = 44100
        output_audio = model.generate(text=args.text, audio_prompt=args.audio_prompt, max_tokens=args.max_tokens, cfg_scale=args.cfg_scale, temperature=args.temperature, top_p=args.top_p)
        print('Audio generation complete.')
        print(f'Saving audio to {args.output}...')
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        sf.write(args.output, output_audio, sample_rate)
        print(f'Audio successfully saved to {args.output}')
    except Exception as e:
        print(f'Error during audio generation or saving: {e}')
        exit(1)

