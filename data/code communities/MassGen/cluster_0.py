# Cluster 0

def main():
    components_dir = Path('components')
    components_dir.mkdir(exist_ok=True)
    print('Extracting components from working presentations...')
    m2l_slides = extract_slides_from_file('m2l.html')
    print(f'Found {len(m2l_slides)} slides in m2l.html')
    head_content = extract_head_section('m2l.html')
    nav_content = extract_navigation_section('m2l.html')
    (components_dir / 'head.html').write_text(head_content)
    print('✅ Saved head.html')
    (components_dir / 'navigation.html').write_text(nav_content)
    print('✅ Saved navigation.html')
    for i, slide in enumerate(m2l_slides, 1):
        title_match = re.search('<!-- Slide \\d+: ([^>]*) -->', slide)
        if title_match:
            title = title_match.group(1).strip()
            filename = clean_slide_title(title)
        else:
            filename = f'slide-{i:02d}'
        slide_content = slide.strip()
        (components_dir / f'{filename}.html').write_text(f'        {slide_content}')
        print(f'✅ Saved {filename}.html')
    print(f'\nExtracted {len(m2l_slides)} slides + head + navigation')
    print('Next: Extract Columbia-specific variants...')

def extract_slides_from_file(file_path):
    """Extract individual slides from a presentation file."""
    content = Path(file_path).read_text()
    slide_pattern = '(<!-- Slide \\d+: [^>]*>.*?)(?=<!-- Slide \\d+:|<!-- Navigation -->|$)'
    slides = re.findall(slide_pattern, content, re.DOTALL)
    return slides

def extract_head_section(file_path):
    """Extract head section from presentation file."""
    content = Path(file_path).read_text()
    head_pattern = '(<!DOCTYPE html>.*?</head>)'
    head_match = re.search(head_pattern, content, re.DOTALL)
    return head_match.group(1) if head_match else ''

def extract_navigation_section(file_path):
    """Extract navigation section from presentation file."""
    content = Path(file_path).read_text()
    nav_pattern = '(<!-- Navigation -->.*?</html>)'
    nav_match = re.search(nav_pattern, content, re.DOTALL)
    return nav_match.group(1) if nav_match else ''

def clean_slide_title(title):
    """Convert slide title to component filename."""
    title = re.sub('<!-- Slide \\d+: ', '', title)
    title = re.sub(' -->', '', title)
    title = title.lower()
    title = re.sub('[^\\w\\s-]', '', title)
    title = re.sub('\\s+', '-', title)
    title = re.sub('-+', '-', title)
    title = title.strip('-')
    return f'slide-{title}'

def test_claude_code_cli_command_building():
    """Test Claude Code CLI command building (without executing) - SKIPPED: File removed."""
    print('🧪 Testing Claude Code CLI command building... SKIPPED (file removed)')
    print('✅ Claude Code CLI command building test skipped')

def test_configuration_files():
    """Test that configuration files are valid."""
    print('🧪 Testing configuration files...')
    import yaml
    config_files = ['massgen/configs/claude_code_cli.yaml', 'massgen/configs/cli_backends_mixed.yaml']
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                assert config is not None, f'Config {config_file} should not be empty'
                print(f'✅ {config_file} is valid')
            except Exception as e:
                print(f'❌ {config_file} is invalid: {e}')
                raise
        else:
            print(f'⚠️  {config_file} not found, skipping')

