# Cluster 61

def convert_csv_to_llm_json(data_dir: str, output_file: str=None) -> str:
    """
    快速转换CSV数据为LLM JSON格式的主函数
    
    Args:
        data_dir (str): 数据目录路径（如 output_300750）
        output_file (str): 输出文件路径（可选）
        
    Returns:
        str: 生成的提示文件路径
        
    Example:
        convert_csv_to_llm_json("output_300750")
        convert_csv_to_llm_json("output_600519", "my_prompt.txt")
    """
    print(f'🔄 开始转换 {data_dir} 中的CSV数据...')
    converter = CSVToLLMConverter(data_dir)
    result_path = converter.save_prompt_to_file(output_file)
    if result_path:
        print(f'✅ 转换完成: {os.path.abspath(result_path)}')
    else:
        print('❌ 转换失败')
    return result_path

def get_stock_data_json(data_dir: str) -> Dict[str, List[Dict]]:
    """
    获取股票数据的JSON格式字典
    
    Args:
        data_dir (str): 数据目录路径（如 output_300750）
        
    Returns:
        Dict[str, List[Dict]]: 包含所有数据的字典
        
    Example:
        data = get_stock_data_json("output_300750")
        print(data.keys())  # 查看所有数据类型
    """
    converter = CSVToLLMConverter(data_dir)
    return converter.get_json_data()

