# Cluster 5

def try_parse_from_embedded_json(text: str) -> list[ModelInfo]:
    models_from_json: list[ModelInfo] = []
    for match in re.finditer('\\{\\s*\\"id\\"\\s*:\\s*\\"', text):
        start = match.start()
        json_str, _end_pos = extract_json_object(text, start)
        if not json_str:
            continue
        if '"host":' not in json_str or '"model":' not in json_str:
            continue
        try:
            data = json.loads(json_str)
        except Exception:
            continue
        name = data.get('name') or (data.get('model') or {}).get('name')
        host_label = data.get('host_label') or ((data.get('host') or {}).get('short_name') or (data.get('host') or {}).get('name'))
        if not name or not host_label:
            continue
        api_id_raw = data.get('slug') or (data.get('model') or {}).get('slug') or name.lower().replace(' ', '-').replace('(', '').replace(')', '')
        host_api_id = data.get('host_api_id')
        api_id = normalize_name_from_slug_or_id(api_id_raw, host_api_id, name)
        context_window = data.get('context_window_tokens') or (data.get('model') or {}).get('context_window_tokens')
        if not context_window:
            formatted = data.get('context_window_formatted') or (data.get('model') or {}).get('contextWindowFormatted')
            context_window = parse_context_window(formatted) if formatted else None
        tool_calling = coalesce_bool(data.get('function_calling'), (data.get('host') or {}).get('function_calling'), (data.get('model') or {}).get('function_calling'))
        structured_outputs = coalesce_bool(data.get('json_mode'), (data.get('host') or {}).get('json_mode'), (data.get('model') or {}).get('json_mode'))
        blended_cost = data.get('price_1m_blended_3_to_1')
        input_cost = data.get('price_1m_input_tokens')
        output_cost = data.get('price_1m_output_tokens')
        timescale = data.get('timescaleData') or {}
        tokens_per_second = timescale.get('median_output_speed') or 0.0
        first_chunk_seconds = timescale.get('median_time_to_first_chunk') or 0.0
        if not tokens_per_second or tokens_per_second <= 0:
            tokens_per_second = 0.1
        if not first_chunk_seconds or first_chunk_seconds <= 0:
            first_chunk_seconds = 0.001
        quality_score = (data.get('model') or {}).get('estimated_intelligence_index') or (data.get('model') or {}).get('intelligence_index') or data.get('estimated_intelligence_index') or data.get('intelligence_index')
        model_info = ModelInfo(name=str(api_id), description=str(name), provider=str(host_label), context_window=int(context_window) if context_window else None, tool_calling=tool_calling, structured_outputs=structured_outputs, metrics=ModelMetrics(cost=ModelCost(blended_cost_per_1m=blended_cost, input_cost_per_1m=input_cost, output_cost_per_1m=output_cost), speed=ModelLatency(time_to_first_token_ms=float(first_chunk_seconds) * 1000.0, tokens_per_second=float(tokens_per_second)), intelligence=ModelBenchmarks(quality_score=float(quality_score) if quality_score else None)))
        models_from_json.append(model_info)
    return models_from_json

def extract_json_object(text: str, start_index: int) -> tuple[Optional[str], int]:
    """Extract a balanced JSON object starting at text[start_index] == '{'.

        Returns (json_string, end_index_after_object) or (None, start_index + 1) if
        no valid object could be parsed.
        """
    if start_index < 0 or start_index >= len(text) or text[start_index] != '{':
        return (None, start_index + 1)
    brace_count = 0
    in_string = False
    escape = False
    i = start_index
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                return (text[start_index:i + 1], i + 1)
        i += 1
    return (None, start_index + 1)

def normalize_name_from_slug_or_id(slug: Optional[str], host_api_id: Optional[str], fallback: str) -> str:
    candidate = host_api_id or slug or fallback
    if not candidate:
        return fallback
    if '/' in candidate:
        candidate = candidate.rsplit('/', 1)[-1]
    return str(candidate)

def parse_context_window(context_str: str) -> int | None:
    """Parse context window strings like '131k', '1m', '128000' to integers."""
    if not context_str:
        return None
    context_str = context_str.strip().lower()
    try:
        if context_str.endswith('k'):
            return int(float(context_str[:-1]) * 1000)
        elif context_str.endswith('m'):
            return int(float(context_str[:-1]) * 1000000)
        else:
            return int(context_str.replace(',', ''))
    except (ValueError, AttributeError):
        return None

def coalesce_bool(*values: Optional[bool | None]) -> Optional[bool]:
    for v in values:
        if isinstance(v, bool):
            return v
    return None

def parse_html_to_models(html_content: str) -> list[ModelInfo]:
    """
    Robustly parse Artificial Analysis model listings.

    Strategy:
    1) First, try to extract embedded JSON objects that the site now renders. These
       contain rich fields like provider, pricing, speed, and latency.
    2) If that fails, fall back to the legacy table-based parser.
    """

    def extract_json_object(text: str, start_index: int) -> tuple[Optional[str], int]:
        """Extract a balanced JSON object starting at text[start_index] == '{'.

        Returns (json_string, end_index_after_object) or (None, start_index + 1) if
        no valid object could be parsed.
        """
        if start_index < 0 or start_index >= len(text) or text[start_index] != '{':
            return (None, start_index + 1)
        brace_count = 0
        in_string = False
        escape = False
        i = start_index
        while i < len(text):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    return (text[start_index:i + 1], i + 1)
            i += 1
        return (None, start_index + 1)

    def coalesce_bool(*values: Optional[bool | None]) -> Optional[bool]:
        for v in values:
            if isinstance(v, bool):
                return v
        return None

    def normalize_name_from_slug_or_id(slug: Optional[str], host_api_id: Optional[str], fallback: str) -> str:
        candidate = host_api_id or slug or fallback
        if not candidate:
            return fallback
        if '/' in candidate:
            candidate = candidate.rsplit('/', 1)[-1]
        return str(candidate)

    def try_parse_from_embedded_json(text: str) -> list[ModelInfo]:
        models_from_json: list[ModelInfo] = []
        for match in re.finditer('\\{\\s*\\"id\\"\\s*:\\s*\\"', text):
            start = match.start()
            json_str, _end_pos = extract_json_object(text, start)
            if not json_str:
                continue
            if '"host":' not in json_str or '"model":' not in json_str:
                continue
            try:
                data = json.loads(json_str)
            except Exception:
                continue
            name = data.get('name') or (data.get('model') or {}).get('name')
            host_label = data.get('host_label') or ((data.get('host') or {}).get('short_name') or (data.get('host') or {}).get('name'))
            if not name or not host_label:
                continue
            api_id_raw = data.get('slug') or (data.get('model') or {}).get('slug') or name.lower().replace(' ', '-').replace('(', '').replace(')', '')
            host_api_id = data.get('host_api_id')
            api_id = normalize_name_from_slug_or_id(api_id_raw, host_api_id, name)
            context_window = data.get('context_window_tokens') or (data.get('model') or {}).get('context_window_tokens')
            if not context_window:
                formatted = data.get('context_window_formatted') or (data.get('model') or {}).get('contextWindowFormatted')
                context_window = parse_context_window(formatted) if formatted else None
            tool_calling = coalesce_bool(data.get('function_calling'), (data.get('host') or {}).get('function_calling'), (data.get('model') or {}).get('function_calling'))
            structured_outputs = coalesce_bool(data.get('json_mode'), (data.get('host') or {}).get('json_mode'), (data.get('model') or {}).get('json_mode'))
            blended_cost = data.get('price_1m_blended_3_to_1')
            input_cost = data.get('price_1m_input_tokens')
            output_cost = data.get('price_1m_output_tokens')
            timescale = data.get('timescaleData') or {}
            tokens_per_second = timescale.get('median_output_speed') or 0.0
            first_chunk_seconds = timescale.get('median_time_to_first_chunk') or 0.0
            if not tokens_per_second or tokens_per_second <= 0:
                tokens_per_second = 0.1
            if not first_chunk_seconds or first_chunk_seconds <= 0:
                first_chunk_seconds = 0.001
            quality_score = (data.get('model') or {}).get('estimated_intelligence_index') or (data.get('model') or {}).get('intelligence_index') or data.get('estimated_intelligence_index') or data.get('intelligence_index')
            model_info = ModelInfo(name=str(api_id), description=str(name), provider=str(host_label), context_window=int(context_window) if context_window else None, tool_calling=tool_calling, structured_outputs=structured_outputs, metrics=ModelMetrics(cost=ModelCost(blended_cost_per_1m=blended_cost, input_cost_per_1m=input_cost, output_cost_per_1m=output_cost), speed=ModelLatency(time_to_first_token_ms=float(first_chunk_seconds) * 1000.0, tokens_per_second=float(tokens_per_second)), intelligence=ModelBenchmarks(quality_score=float(quality_score) if quality_score else None)))
            models_from_json.append(model_info)
        return models_from_json
    json_models = try_parse_from_embedded_json(html_content)
    if json_models:
        console.print(f'[bold blue]Parsed {len(json_models)} models from embedded JSON[/bold blue]')
    soup = BeautifulSoup(html_content, 'html.parser')
    models: list[ModelInfo] = []
    headers = [th.get_text(strip=True) for th in soup.find_all('th')]
    console.print(f'[bold blue]Found {len(headers)} headers[/bold blue]')
    rows = soup.find_all('tr')

    def is_data_row(tr) -> bool:
        tds = tr.find_all('td')
        return len(tds) >= 6
    rows = [r for r in rows if is_data_row(r)]
    console.print(f'[bold green]Processing {len(rows)} models...[/bold green]')

    def parse_price_tokens_latency(cells: list[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        price = None
        tokens_per_s = None
        latency_s = None
        price_idx = None
        for idx, txt in enumerate(cells):
            if '$' in txt:
                try:
                    price = float(txt.replace('$', '').replace(',', '').strip())
                    price_idx = idx
                    break
                except Exception:
                    continue
        if price_idx is not None:
            found = []
            for txt in cells[price_idx + 1:price_idx + 6]:
                try:
                    val = float(txt.replace(',', '').strip())
                    found.append(val)
                except Exception:
                    continue
                if len(found) >= 2:
                    break
            if len(found) >= 2:
                tokens_per_s, latency_s = (found[0], found[1])
        return (price, tokens_per_s, latency_s)
    for row in track(rows, description='Parsing models...'):
        cells_el = row.find_all('td')
        cells = [c.get_text(strip=True) for c in cells_el]
        if not cells:
            continue
        try:
            provider_img = cells_el[0].find('img')
            provider = provider_img['alt'].replace(' logo', '') if provider_img else 'Unknown'
            model_name_elem = cells_el[1].find('span')
            if model_name_elem:
                display_name = model_name_elem.text.strip()
            else:
                display_name = cells[1].strip()
            href = None
            link = row.find('a', href=re.compile('/models/'))
            if link and link.has_attr('href'):
                href = link['href']
            api_id = None
            if href:
                api_id = href.rstrip('/').rsplit('/', 1)[-1]
            if not api_id:
                api_id = display_name.lower().replace(' ', '-').replace('(', '').replace(')', '').replace('/', '-')
            context_window_text = cells[2]
            context_window = parse_context_window(context_window_text)
            tool_calling = None
            structured_outputs = None
            quality_score = None
            for txt in cells:
                if txt.endswith('%'):
                    try:
                        quality_score = float(txt.replace('%', '').strip())
                        break
                    except Exception:
                        pass
            blended_cost, tokens_per_sec, latency_sec = parse_price_tokens_latency(cells)
            if tokens_per_sec is None:
                tokens_per_sec = 0.1
            if latency_sec is None:
                latency_sec = 0.001
            model_info = ModelInfo(name=api_id, description=display_name, provider=provider, context_window=context_window, tool_calling=tool_calling, structured_outputs=structured_outputs, metrics=ModelMetrics(cost=ModelCost(blended_cost_per_1m=blended_cost), speed=ModelLatency(time_to_first_token_ms=float(latency_sec) * 1000.0, tokens_per_second=float(tokens_per_sec)), intelligence=ModelBenchmarks(quality_score=quality_score)))
            models.append(model_info)
        except Exception as e:
            console.print(f'[red]Error processing row: {e}[/red]')
            console.print(f'[yellow]Row content: {str(row)}[/yellow]')
            continue
    if json_models:
        merged: dict[tuple[str, str], ModelInfo] = {}
        for m in json_models:
            merged[m.provider.lower(), m.name.lower()] = m
        for m in models:
            key = (m.provider.lower(), m.name.lower())
            if key not in merged:
                merged[key] = m
        return list(merged.values())
    return models

def is_data_row(tr) -> bool:
    tds = tr.find_all('td')
    return len(tds) >= 6

def parse_price_tokens_latency(cells: list[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    price = None
    tokens_per_s = None
    latency_s = None
    price_idx = None
    for idx, txt in enumerate(cells):
        if '$' in txt:
            try:
                price = float(txt.replace('$', '').replace(',', '').strip())
                price_idx = idx
                break
            except Exception:
                continue
    if price_idx is not None:
        found = []
        for txt in cells[price_idx + 1:price_idx + 6]:
            try:
                val = float(txt.replace(',', '').strip())
                found.append(val)
            except Exception:
                continue
            if len(found) >= 2:
                break
        if len(found) >= 2:
            tokens_per_s, latency_s = (found[0], found[1])
    return (price, tokens_per_s, latency_s)

@app.command()
def main(input_file: Path=typer.Argument(..., help='Path to the HTML file containing the benchmark table', exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), output_file: Path=typer.Argument('src/mcp_agent/data/artificial_analysis_llm_benchmarks.json', help='Path to the output JSON file', resolve_path=True)):
    """
    Parse LLM benchmark HTML tables from Artificial Analysis and convert to JSON.
    """
    console.print(f'[bold]Reading HTML from:[/bold] {input_file}')
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        models = parse_html_to_models(html_content)
        if not models:
            console.print('[red]No models found in the HTML file![/red]')
            raise typer.Exit(1)
        console.print(f'\n[bold green]Successfully parsed {len(models)} models![/bold green]\n')
        display_summary(models)
        export_to_json(models, str(output_file))
        console.print(f'\n[bold]Output saved to:[/bold] {output_file}')
    except Exception as e:
        console.print(f'[red]Error: {e}[/red]')
        raise typer.Exit(1)

def display_summary(models: list[ModelInfo]):
    """Display a summary table of parsed models."""
    table = Table(title=f'Parsed Models Summary ({len(models)} models)')
    table.add_column('#', style='dim', width=3)
    table.add_column('Provider', style='cyan', no_wrap=True)
    table.add_column('Model', style='magenta', max_width=50)
    table.add_column('Context', justify='right', style='green')
    table.add_column('Tools', justify='center')
    table.add_column('JSON', justify='center')
    table.add_column('Quality', justify='right', style='yellow')
    table.add_column('Cost/1M', justify='right', style='red')
    table.add_column('Speed', justify='right', style='blue')
    for idx, model in enumerate(models, 1):
        model_name = model.description or model.name
        if len(model_name) > 50:
            model_name = model_name[:47] + '...'
        table.add_row(str(idx), model.provider, model_name, f'{model.context_window:,}' if model.context_window else 'N/A', '✓' if model.tool_calling else '✗' if model.tool_calling is False else '?', '✓' if model.structured_outputs else '✗' if model.structured_outputs is False else '?', f'{model.metrics.intelligence.quality_score:.1f}%' if model.metrics.intelligence.quality_score else 'N/A', f'${model.metrics.cost.blended_cost_per_1m:.2f}' if model.metrics.cost.blended_cost_per_1m else 'N/A', f'{model.metrics.speed.tokens_per_second:.0f} t/s' if model.metrics.speed.tokens_per_second else 'N/A')
    console.print(table)

def export_to_json(models: list[ModelInfo], output_file: str='model_benchmarks5.json'):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([m.model_dump() for m in models], f, indent=2)

