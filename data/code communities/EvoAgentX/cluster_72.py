# Cluster 72

def main():
    """Main function to run all file system examples"""
    print('===== FILE SYSTEM TOOLS EXAMPLES =====')
    run_advanced_file_operations()
    print('\n===== ALL FILE SYSTEM EXAMPLES COMPLETED =====')

def run_advanced_file_operations():
    """
    Run examples demonstrating advanced file operations and format handling.
    """
    print('\n===== ADVANCED FILE OPERATIONS =====\n')
    try:
        storage_toolkit = StorageToolkit()
        save_tool = storage_toolkit.get_tool('save')
        read_tool = storage_toolkit.get_tool('read')
        print('1. Testing CSV file operations...')
        csv_content = 'name,age,city\nJohn Doe,30,New York\nJane Smith,25,Los Angeles\nBob Johnson,35,Chicago'
        csv_result = save_tool(file_path='sample_data.csv', content=csv_content)
        if csv_result.get('success'):
            print('✓ CSV file saved successfully')
            csv_read_result = read_tool(file_path='sample_data.csv')
            if csv_read_result.get('success'):
                print('✓ CSV file read successfully')
                print(f'Content: {csv_read_result.get('content', '')[:100]}...')
            else:
                print(f'❌ Failed to read CSV file: {csv_read_result.get('error')}')
        else:
            print(f'❌ Failed to save CSV file: {csv_result.get('error')}')
        print('\n2. Testing YAML file operations...')
        yaml_content = 'name: Sample YAML\nversion: 1.0\nfeatures:\n  - feature1\n  - feature2\nmetadata:\n  author: Test User\n  date: 2024-01-01'
        yaml_result = save_tool(file_path='sample_config.yaml', content=yaml_content)
        if yaml_result.get('success'):
            print('✓ YAML file saved successfully')
            yaml_read_result = read_tool(file_path='sample_config.yaml')
            if yaml_read_result.get('success'):
                print('✓ YAML file read successfully')
                print(f'Content: {yaml_read_result.get('content', '')[:100]}...')
            else:
                print(f'❌ Failed to read YAML file: {yaml_read_result.get('error')}')
        else:
            print(f'❌ Failed to save YAML file: {yaml_result.get('error')}')
        print('\n3. Testing PDF file operations...')
        pdf_content = "Test PDF Document\n\nThis is a test PDF created by EvoAgentX.\n\nFeatures:\n• PDF creation from text\n• Automatic formatting\n• Professional layout\n\nThis demonstrates the storage system's PDF capabilities."
        pdf_result = save_tool(file_path='test_pdf.pdf', content=pdf_content)
        if pdf_result.get('success'):
            print('✓ PDF file created successfully')
        else:
            print(f'❌ Failed to create PDF file: {pdf_result.get('error')}')
        pdf_read_result = read_tool(file_path='test_pdf.pdf')
        if pdf_read_result.get('success'):
            print('✓ PDF file read successfully')
            print(f'Content: {pdf_read_result.get('content', '')[:100]}...')
        else:
            print(f'❌ Failed to read PDF file: {pdf_read_result.get('error')}')
        print('\n4. Testing file deletion...')
        delete_tool = storage_toolkit.get_tool('delete')
        test_files = ['sample_document.txt', 'sample_data.json', 'custom_test.txt']
        for test_file in test_files:
            if os.path.exists(test_file):
                delete_result = delete_tool(path=test_file)
                if delete_result.get('success'):
                    print(f'✓ Deleted {test_file}')
                else:
                    print(f'❌ Failed to delete {test_file}: {delete_result.get('error')}')
        print('\n✓ Advanced file operations completed')
    except Exception as e:
        print(f'Error running advanced file operations: {str(e)}')

