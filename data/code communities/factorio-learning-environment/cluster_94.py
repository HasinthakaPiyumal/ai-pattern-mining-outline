# Cluster 94

class TestPythonParser:

    def test_single_valid_python_block(self):
        """Test that a single block of valid Python code is returned unchanged"""
        code = '\nx = 1\ny = 2\nprint(x + y)\n'
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(code)))
        assert result.strip() == code.strip()

    def test_mixed_content_with_explanation(self):
        """Test handling of mixed content with explanation and code"""
        content = "\nLet's create a loop to process items\n\nfor i in range(10):\n    print(i)\n\nThis will print numbers 0-9\n"
        expected = "# Let's create a loop to process items\n\nfor i in range(10):\n    print(i)\n\n# This will print numbers 0-9"
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(content)))
        assert result.strip() == expected.strip()

    def test_complex_pipe_example_1(self):
        """Test the first complex pipe placement example"""
        content = 'The repeated attempts to place pipes indicate that the terrain or path chosen is not suitable for pipe placement. Let\'s try a different approach by moving further away from the current path and using a combination of underground and regular pipes to bypass any potential obstacles.\n\n# Define positions for offshore pump and boiler\noffshore_pump_position = Position(x=-11.5, y=26.5)\nboiler_position = Position(x=-3.5, y=26.0)\n\n# Move to offshore pump position\nmove_to(offshore_pump_position)\n\n# Inspect and clear obstructions manually along a new path, slightly above the previous attempt\npipe_path_positions = [Position(x=i, y=28.5) for i in range(-11, -3)]\nfor pos in pipe_path_positions:\n    move_to(pos)\n    entities = get_entities(position=pos, radius=0.5)\n    print(f"Entities at {pos}: {entities}")\n    for entity in entities:\n        if not isinstance(entity, Pipe):\n            pickup_entity(entity)\n            print(f"Cleared obstruction at {entity.position}.")'
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(content)))
        assert '# The repeated attempts' in result
        assert 'offshore_pump_position = Position(x=-11.5, y=26.5)' in result
        assert 'pipe_path_positions = [Position(x=i, y=28.5) for i in range(-11, -3)]' in result

    def test_complex_pipe_example_2(self):
        """Test the second complex pipe placement example"""
        content = "The current approach of trying to place pipes at various y-coordinates has not been successful. Let's try a different strategy by placing the pipes directly from the offshore pump to the boiler without skipping positions, ensuring that we use both underground and regular pipes where necessary.\n\n# Define positions for offshore pump and boiler\noffshore_pump_position = Position(x=-11.5, y=26.5)\nboiler_position = Position(x=-3.5, y=26.0)\n\n# Move to offshore pump position\nmove_to(offshore_pump_position)"
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(content)))
        assert '# The current approach' in result
        assert 'offshore_pump_position = Position(x=-11.5, y=26.5)' in result
        assert 'move_to(offshore_pump_position)' in result

    def test_empty_content(self):
        """Test handling of empty content"""
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage('')))
        assert not result

    def test_only_comments(self):
        """Test handling of content that contains only comments"""
        content = '\nThis is just some text\nwith multiple lines\nbut no actual code\n'
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(content)))
        assert result.startswith('"""') or result.startswith('#')

    def test_invalid_python(self):
        """Test handling of invalid Python syntax"""
        content = '\nfor i in range(10)    # Missing colon\n    print(i)\n'
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(content)))
        assert result.startswith('"""') or result.startswith('#')

    def test_markdown_code_blocks(self):
        """Test handling of markdown code blocks"""
        content = "\nHere's some explanation\n\n```python\nx = 1\ny = 2\nprint(x + y)\n```\n\nMore explanation here\n"
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(content)))
        assert 'x = 1' in result

    def test_mixed_valid_invalid_blocks(self):
        """Test handling of mixed valid and invalid Python blocks"""
        content = '\ndef valid_function():\n    return 42\n\nThis is some invalid content\n\nx = 1\ny = 2\nprint(x + y)\n'
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(content)))
        assert 'def valid_function():' in result
        assert 'return 42' in result
        assert '# This is some invalid content' in result
        assert 'x = 1' in result

    def test_indentation_preservation(self):
        """Test that indentation is preserved in valid Python blocks"""
        content = "\ndef nested_function():\n    if True:\n        for i in range(10):\n            print(i)\n            if i % 2 == 0:\n                print('even')\n"
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(content)))
        assert '            print(i)' in result
        assert "                print('even')" in result

    def test_multiline_string_preservation(self):
        """Test that multiline strings in valid Python are preserved"""
        content = '\ntext = """\nThis is a\nmultiline string\nin valid Python\n"""\nprint(text)\n'
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(content)))
        assert 'text = """' in result
        assert 'multiline string' in result

    def test_mixed_comments_and_code(self):
        """Test handling of mixed inline comments and code"""
        content = '\n# This is a comment\nx = 1  # Inline comment\n# Another comment\ny = 2\n'
        result, _ = PythonParser.extract_code(MockLLMResponse(MockMessage(content)))
        assert '# This is a comment' in result
        assert 'x = 1  # Inline comment' in result
        assert '# Another comment' in result
        assert 'y = 2' in result

