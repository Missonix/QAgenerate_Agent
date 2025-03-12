
"""
电商商品QA对生成系统启动脚本
"""

import os
import sys
import logging
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qa_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查依赖项"""
    required_modules = [
        "langchain_openai", 
        "langgraph", 
        "pandas", 
        "openpyxl", 
        "aiofiles"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("缺少必要的依赖项:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\n请安装缺失的依赖项:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    return True

def check_required_files():
    """检查必要的文件"""
    required_files = [
        "product_data_processor.py",
        "async_qa_generator.py",
        "qa_agent_simple.py"  
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("缺少必要的系统文件:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='启动电商商品QA对生成系统')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()
    
    # 检查依赖项和文件
    if not check_dependencies() or not check_required_files():
        sys.exit(1)
    
    # 检查并清理输出文件以确保新的运行体验
    clean_files = ["products_data.json", "qa_output.json"]
    for file in clean_files:
        if os.path.exists(file):
            logger.info(f"清理之前的输出文件: {file}")
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"文件 {file} 内容大小: {len(content)} 字节")
            except Exception as e:
                logger.warning(f"读取文件 {file} 时出错: {str(e)}")
    
    print("=" * 60)
    print("欢迎使用电商商品QA对生成系统")
    print("=" * 60)
    print("本系统将帮助您基于商品信息生成自然、多样化的问答对")
    print("您可以提供TXT、DOCX、XLSX、CSV、JSON文件，或直接输入商品数据")
    print("=" * 60)
    
    # 显示系统状态
    print("电商商品QA对生成系统已启动，正在初始化...")
    
    try:
        # 检查示例文件是否存在
        example_files = ["example_product.txt", "test_product.txt"]
        available_examples = [f for f in example_files if os.path.exists(f)]
        if available_examples:
            logger.info(f"发现可用的示例文件: {available_examples}")
        
        # 导入QA代理系统
        from qa_agent_simple import main as run_qa_system
        logger.info("成功导入QA代理系统")
        
        # 启动系统
        run_qa_system()
    except Exception as e:
        logger.error(f"运行系统时出错: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"运行系统时出错: {str(e)}")
            print("如需查看详细错误信息，请使用 --debug 选项重新启动")
    
    print("\n感谢使用电商商品QA对生成系统！")

if __name__ == "__main__":
    main() 