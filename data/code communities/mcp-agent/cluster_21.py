# Cluster 21

def _resolve_instruction_arg(instruction: Optional[str]) -> Optional[str]:
    if not instruction:
        return None
    try:
        if instruction.startswith('text:'):
            return instruction[len('text:'):]
        if instruction.startswith('http://') or instruction.startswith('https://'):
            try:
                import httpx
                r = httpx.get(instruction, timeout=10.0)
                r.raise_for_status()
                return r.text
            except Exception:
                try:
                    from urllib.request import urlopen
                    with urlopen(instruction, timeout=10) as resp:
                        return resp.read().decode('utf-8')
                except Exception as e:
                    raise typer.Exit(6) from e
        p = Path(instruction).expanduser()
        if p.exists() and p.is_file():
            return p.read_text(encoding='utf-8')
        return instruction
    except Exception:
        return instruction

