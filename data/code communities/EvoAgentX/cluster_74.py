# Cluster 74

def main():
    """Main function to run MCP examples"""
    print('===== MCP TOOL INTEGRATION EXAMPLES =====\n')
    run_mcp_example()
    print('\n===== ALL MCP TOOL EXAMPLES COMPLETED =====')

def run_mcp_example():
    """
    Run a simple example using the MCP toolkit to search for research papers about 'artificial intelligence'.
    This uses the sample_mcp.config file to configure the arXiv MCP client.
    """
    print('\n===== MCP TOOLKIT EXAMPLE (arXiv) =====\n')
    config_path = os.path.join(os.getcwd(), 'examples', 'tools', 'sample_mcp.config')
    print(f'Loading MCP configuration from: {config_path}')
    try:
        mcp_toolkit = MCPToolkit(config_path=config_path)
        toolkits = mcp_toolkit.get_toolkits()
        print(f'✓ Connected to {len(toolkits)} MCP server(s)')
        arxiv_tool = None
        for toolkit_item in toolkits:
            for tool in toolkit_item.get_tools():
                print(f'Tool: {tool.name}')
                print(f'Description: {tool.description}')
                print('-' * 30)
                if 'search' in tool.name.lower() or 'arxiv' in tool.name.lower():
                    arxiv_tool = tool
                    break
            if arxiv_tool:
                break
        if arxiv_tool:
            print(f'✓ Found arXiv tool: {arxiv_tool.name}')
            print(f'  Description: {arxiv_tool.description}')
            search_query = 'artificial intelligence'
            print(f"\nSearching for research papers about: '{search_query}'")
            result = arxiv_tool(**{'query': search_query})
            print('\nSearch Results:')
            print('-' * 50)
            print(f'Result type: {type(result)}')
            print(f'Result content: {result}')
            print('-' * 50)
            print('✓ MCP arXiv example completed')
        else:
            print('❌ No suitable arXiv search tool found in the MCP configuration.')
    except Exception as e:
        print(f'❌ Error running MCP example: {str(e)}')
        print('Make sure the arXiv MCP server is properly configured and running.')
    finally:
        if 'mcp_toolkit' in locals():
            mcp_toolkit.disconnect()
            print('✓ MCP connection closed')

