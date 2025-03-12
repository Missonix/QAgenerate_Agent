import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入LangChain组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 导入数据处理模块
try:
    from product_data_processor import ProductDataProcessor
except ImportError:
    logger.error("未找到product_data_processor.py模块，请确保该文件在当前目录中")
    sys.exit(1)

# 尝试导入QA生成模块
try:
    from async_qa_generator import main as generate_qa
except ImportError:
    logger.error("未找到async_qa_generator.py模块，请确保该文件在当前目录中")
    sys.exit(1)

# 初始化大语言模型
model = ChatOpenAI(
    api_key="",
    base_url="",
    model="",
    temperature=0.7
)

# 会话状态类
class SessionState:
    """会话状态类"""
    def __init__(self):
        self.messages = []  # 对话历史
        self.input_source = None  # 输入源（文件路径或文本）
        self.input_type = None  # 输入类型 (file/text)
        self.data_processed = False  # 数据是否已处理
        self.products_file = None  # 处理后的商品数据文件
        self.qa_count = None  # 要生成的QA对数量
        self.qa_generation_started = False  # 是否已开始生成QA对
        self.qa_file = None  # 生成的QA对文件
        self.workflow_completed = False  # 工作流是否已完成

# 创建自定义的工具执行器
class ToolExecutor:
    """工具执行器，用于执行函数工具"""
    def __init__(self, tools):
        self.tools = {tool["name"]: tool["function"] for tool in tools}
    
    def execute(self, tool_name, tool_input):
        if tool_name not in self.tools:
            raise ValueError(f"未找到工具 '{tool_name}'")
        try:
            return self.tools[tool_name](**tool_input)
        except Exception as e:
            logger.error(f"执行工具 '{tool_name}' 时出错: {str(e)}")
            raise

# 工具定义
def validate_product_input(
    input_source: str,
    input_type: str = "auto"
) -> Dict[str, Any]:
    """
    验证商品信息输入是否规范
    
    参数:
    - input_source: 输入源（文件路径或文本内容）
    - input_type: 输入类型 ('auto', 'file', 'text')
    
    返回:
    - 验证结果字典
    """
    logger.info(f"正在验证输入: {input_source[:100]}...")
    
    # 确定输入类型
    actual_input_type = input_type
    if input_type == 'auto':
        if os.path.exists(input_source):
            actual_input_type = 'file'
        else:
            actual_input_type = 'text'
    
    # 初始化数据处理器
    processor = ProductDataProcessor()
    
    # 验证输入
    try:
        if actual_input_type == 'file':
            # 检查文件是否存在
            if not os.path.exists(input_source):
                return {
                    "is_valid": False,
                    "reason": f"文件不存在: {input_source}",
                    "input_type": actual_input_type,
                    "format_guide": get_format_guide()
                }
            
            # 检查文件扩展名
            _, ext = os.path.splitext(input_source)
            ext = ext.lower()
            
            if ext not in processor.supported_extensions:
                return {
                    "is_valid": False,
                    "reason": f"不支持的文件格式: {ext}。支持的格式: {', '.join(processor.supported_extensions)}",
                    "input_type": actual_input_type,
                    "format_guide": get_format_guide()
                }
            
            # 尝试加载和解析少量数据
            try:
                # 根据文件类型执行不同的验证
                if ext == '.txt':
                    with open(input_source, 'r', encoding='utf-8') as f:
                        content = f.read(2000)  # 只读取前2000个字符验证
                    sample_products = processor._process_txt_content(content)
                elif ext == '.docx':
                    if not hasattr(processor, '_process_docx_file'):
                        return {
                            "is_valid": False,
                            "reason": "系统未安装python-docx库，无法处理Word文档",
                            "input_type": actual_input_type,
                            "format_guide": get_format_guide()
                        }
                    # Word文档验证较复杂，这里假设它有效
                    sample_products = [{"is_sample": True}]
                elif ext in ['.xlsx', '.csv']:
                    # Excel/CSV验证较复杂，这里假设它有效
                    sample_products = [{"is_sample": True}]
                elif ext == '.json':
                    with open(input_source, 'r', encoding='utf-8') as f:
                        try:
                            json_data = json.load(f)
                            if isinstance(json_data, list):
                                sample_products = json_data[:2]  # 只取前两个用于验证
                            elif isinstance(json_data, dict):
                                sample_products = [json_data]
                            else:
                                return {
                                    "is_valid": False,
                                    "reason": "JSON文件格式错误，必须是对象或对象数组",
                                    "input_type": actual_input_type,
                                    "format_guide": get_format_guide()
                                }
                        except json.JSONDecodeError:
                            return {
                                "is_valid": False,
                                "reason": "JSON解析错误，请检查JSON格式是否正确",
                                "input_type": actual_input_type,
                                "format_guide": get_format_guide()
                            }
                else:
                    sample_products = []
                
                if not sample_products:
                    return {
                        "is_valid": False,
                        "reason": "无法从输入中解析出商品信息",
                        "input_type": actual_input_type,
                        "format_guide": get_format_guide()
                    }
                
                # 验证成功
                return {
                    "is_valid": True,
                    "input_type": actual_input_type,
                    "message": f"输入格式有效，已检测到商品信息"
                }
                
            except Exception as e:
                return {
                    "is_valid": False,
                    "reason": f"解析错误: {str(e)}",
                    "input_type": actual_input_type,
                    "format_guide": get_format_guide()
                }
        else:  # 文本输入
            # 尝试解析文本
            try:
                sample_products = processor.process_text(input_source[:2000])  # 只处理前2000个字符用于验证
                
                if not sample_products:
                    return {
                        "is_valid": False,
                        "reason": "无法从文本中解析出商品信息",
                        "input_type": actual_input_type,
                        "format_guide": get_format_guide()
                    }
                
                # 验证成功
                return {
                    "is_valid": True,
                    "input_type": actual_input_type,
                    "message": f"输入格式有效，已检测到商品信息"
                }
                
            except Exception as e:
                return {
                    "is_valid": False,
                    "reason": f"解析错误: {str(e)}",
                    "input_type": actual_input_type,
                    "format_guide": get_format_guide()
                }
    
    except Exception as e:
        return {
            "is_valid": False,
            "reason": f"验证过程中发生错误: {str(e)}",
            "input_type": actual_input_type,
            "format_guide": get_format_guide()
        }

def process_product_data(
    input_source: str,
    input_type: str = "auto",
    output_json: str = "products_data.json"
) -> Dict[str, Any]:
    """
    处理商品数据并转换为标准JSON
    
    参数:
    - input_source: 输入源（文件路径或文本内容）
    - input_type: 输入类型 ('auto', 'file', 'text')
    - output_json: 输出JSON文件路径
    
    返回:
    - 处理结果字典
    """
    logger.info(f"正在处理商品数据: {input_source[:50]}...")
    
    # 确定输入类型
    actual_input_type = input_type
    if input_type == 'auto':
        if os.path.exists(input_source):
            actual_input_type = 'file'
        else:
            actual_input_type = 'text'
    
    # 确保输出文件路径是绝对路径
    output_json_abs = os.path.abspath(output_json)
    logger.info(f"输出JSON文件绝对路径: {output_json_abs}")
    
    # 初始化数据处理器
    processor = ProductDataProcessor()
    logger.info(f"已初始化ProductDataProcessor，开始处理{actual_input_type}...")
    
    # 处理输入
    try:
        if actual_input_type == 'file':
            logger.info(f"正在处理文件: {input_source}")
            try:
                with open(input_source, 'r', encoding='utf-8') as f:
                    sample_content = f.read(200)
                logger.info(f"文件内容示例: {sample_content}...")
            except Exception as e:
                logger.warning(f"读取文件内容示例时出错: {str(e)}")
                
            products = processor.process_file(input_source)
            logger.info(f"文件处理完成，解析到 {len(products)} 个商品")
        else:
            logger.info(f"正在处理文本内容")
            products = processor.process_text(input_source)
            logger.info(f"文本处理完成，解析到 {len(products)} 个商品")
        
        if not products:
            logger.warning("未找到有效的商品数据")
            return {
                "success": False,
                "reason": "未找到有效的商品数据",
                "output_file": None
            }
        
        # 详细记录商品信息
        for i, product in enumerate(products):
            product_id = product.get('id', product.get('商品ID', f'未知ID_{i}'))
            product_name = product.get('商品名称', product.get('name', '未知商品'))
            logger.info(f"商品 {i+1}: ID={product_id}, 名称={product_name}")
        
        # 保存为JSON，确保写入磁盘
        output_path = processor.save_to_json(products, output_json)
        logger.info(f"已将 {len(products)} 个商品保存到 {output_path}")
        
        # 验证文件是否成功写入
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"已确认输出文件存在，大小为 {file_size} 字节")
            
            # 读取部分内容作为验证
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    start_content = f.read(200)
                logger.info(f"输出文件内容开始部分: {start_content}...")
            except Exception as e:
                logger.warning(f"读取输出文件内容示例时出错: {str(e)}")
        else:
            logger.error(f"无法确认输出文件 {output_path} 存在")
            return {
                "success": False,
                "reason": f"保存成功但无法确认输出文件 {output_path} 存在",
                "output_file": output_path
            }
        
        return {
            "success": True,
            "message": f"商品数据处理完成，已保存 {len(products)} 个商品到 {output_path}",
            "output_file": output_path,
            "product_count": len(products)
        }
        
    except Exception as e:
        logger.error(f"处理商品数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "reason": f"处理数据时出错: {str(e)}",
            "output_file": None
        }

def generate_qa_pairs(
    json_file: str,
    num_pairs: int = 5,
    output_file: str = "qa_output.json",
    concurrency: int = 3
) -> Dict[str, Any]:
    """
    生成QA对
    
    参数:
    - json_file: 商品数据JSON文件路径
    - num_pairs: 每个商品生成的QA对数量
    - output_file: 输出文件路径
    - concurrency: 并发请求数量
    
    返回:
    - 生成结果字典
    """
    logger.info(f"开始生成QA对，从 {json_file} 生成 {num_pairs} 对/商品，并发数: {concurrency}")
    
    # 确保输入和输出文件的绝对路径
    json_file_abs = os.path.abspath(json_file)
    output_file_abs = os.path.abspath(output_file)
    
    logger.info(f"输入文件绝对路径: {json_file_abs}")
    logger.info(f"输出文件绝对路径: {output_file_abs}")
    
    try:
        # 检查商品数据文件是否存在
        if not os.path.exists(json_file_abs):
            error_msg = f"找不到商品数据文件: {json_file_abs}"
            logger.error(error_msg)
            return {
                "success": False,
                "reason": error_msg,
                "output_file": None
            }
        
        # 验证输入文件内容
        try:
            with open(json_file_abs, 'r', encoding='utf-8') as f:
                products = json.load(f)
                file_size = os.path.getsize(json_file_abs)
                logger.info(f"成功打开商品数据文件，大小: {file_size} 字节，包含 {len(products)} 个商品")
                
                # 显示部分内容
                sample_json = json.dumps(products[:2] if len(products) > 1 else products, ensure_ascii=False)[:500]
                logger.info(f"文件内容示例: {sample_json}...")
        except json.JSONDecodeError as e:
            error_msg = f"商品数据文件不是有效的JSON格式: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "reason": error_msg,
                "output_file": None
            }
        except Exception as e:
            error_msg = f"读取商品数据文件时出错: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "reason": error_msg,
                "output_file": None
            }
        
        if not products:
            error_msg = "商品数据为空"
            logger.error(error_msg)
            return {
                "success": False,
                "reason": error_msg,
                "output_file": None
            }
        
        logger.info(f"成功加载商品数据，共 {len(products)} 个商品")
        
        # 记录商品ID列表
        product_ids = []
        for product in products:
            product_id = product.get('id', product.get('商品ID', None))
            if product_id:
                product_ids.append(product_id)
        
        logger.info(f"商品ID列表: {product_ids}")
        
        try:
            # 导入并调用异步QA生成器
            logger.info("正在导入异步QA生成器...")
            from async_qa_generator import main as generate_qa
            logger.info("成功导入异步QA生成器")
            
            # 调用QA生成函数
            logger.info(f"开始生成QA对，将为 {len(products)} 个商品生成 {num_pairs} 对/商品，" 
                       f"总计 {len(products) * num_pairs} 对QA")
            
            # 清理旧的输出文件
            if os.path.exists(output_file_abs):
                logger.info(f"删除旧的输出文件: {output_file_abs}")
                try:
                    os.remove(output_file_abs)
                except Exception as e:
                    logger.warning(f"删除旧输出文件时出错: {str(e)}")
            
            # 实际调用异步生成函数
            logger.info(f"调用异步生成函数，参数: json_file={json_file_abs}, product_ids={None}, "
                       f"num_pairs={num_pairs}, output_file={output_file_abs}, concurrency={concurrency}")
            
            # 确认目标目录存在
            output_dir = os.path.dirname(output_file_abs)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"已创建输出目录: {output_dir}")
            
            qa_pairs = generate_qa(
                json_file_abs, 
                None,  # 处理所有商品
                num_pairs, 
                output_file_abs, 
                concurrency
            )
            
            logger.info(f"QA生成函数调用完成，返回了 {len(qa_pairs) if qa_pairs else 0} 对QA")
            
        except ImportError as e:
            error_msg = f"导入异步QA生成器失败: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "reason": error_msg,
                "output_file": None
            }
        except Exception as e:
            error_msg = f"QA生成过程中发生错误: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "reason": error_msg,
                "output_file": None
            }
        
        # 检查生成结果
        if os.path.exists(output_file_abs):
            try:
                with open(output_file_abs, 'r', encoding='utf-8') as f:
                    qa_pairs = json.load(f)
                    file_size = os.path.getsize(output_file_abs)
                
                logger.info(f"成功生成 {len(qa_pairs)} 对QA，已保存到 {output_file_abs}，文件大小: {file_size} 字节")
                
                # 记录前几个QA对示例
                if qa_pairs:
                    sample_size = min(2, len(qa_pairs))
                    for i in range(sample_size):
                        product_id = qa_pairs[i].get('id', '未知ID')
                        logger.info(f"QA示例 {i+1}: 商品ID={product_id}")
                        logger.info(f"QA示例 {i+1}: Q: {qa_pairs[i]['question']}")
                        answer_sample = qa_pairs[i]['answer'][:100] + ('...' if len(qa_pairs[i]['answer']) > 100 else '')
                        logger.info(f"QA示例 {i+1}: A: {answer_sample}")
                
                # 检查生成的QA对应的商品ID是否在原始数据中
                qa_product_ids = set(qa_pair.get('id') for qa_pair in qa_pairs if 'id' in qa_pair)
                missing_ids = [id for id in qa_product_ids if id not in product_ids]
                
                if missing_ids:
                    logger.warning(f"发现 {len(missing_ids)} 个QA对的商品ID不在原始数据中: {missing_ids[:5]}")
                
                return {
                    "success": True,
                    "message": f"QA对生成完成，已生成 {len(qa_pairs)} 对QA",
                    "output_file": output_file_abs,
                    "qa_count": len(qa_pairs)
                }
            except json.JSONDecodeError as e:
                error_msg = f"生成的QA文件不是有效的JSON格式: {str(e)}"
                logger.error(error_msg)
                
                # 尝试读取部分内容进行问题诊断
                try:
                    with open(output_file_abs, 'r', encoding='utf-8') as f:
                        content = f.read(500)
                    logger.error(f"生成的QA文件内容示例: {content}...")
                except Exception:
                    pass
                
                return {
                    "success": False,
                    "reason": error_msg,
                    "output_file": output_file_abs
                }
            except Exception as e:
                error_msg = f"读取生成的QA对时出错: {str(e)}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "reason": error_msg,
                    "output_file": output_file_abs
                }
        else:
            error_msg = f"QA对生成失败，未找到输出文件 {output_file_abs}"
            logger.error(error_msg)
            return {
                "success": False,
                "reason": error_msg,
                "output_file": None
            }
            
    except Exception as e:
        error_msg = f"生成QA对过程中发生未知错误: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "reason": error_msg,
            "output_file": None
        }

def get_format_guide() -> str:
    """获取输入格式指南"""
    guide = """
## 商品数据输入格式指南

### TXT格式 (键值对)
```
商品ID: p001
商品名称: 专业无线降噪耳机
价格: ¥1299
品牌: SoundMaster

规格:
- 蓝牙版本: 5.2
- 电池续航: 30小时

---

商品ID: p002
商品名称: 智能手表
价格: ¥899
...
```

### JSON格式
```json
[
  {
    "id": "p001",
    "name": "专业无线降噪耳机",
    "price": "¥1299",
    "brand": "SoundMaster",
    "specs": {
      "蓝牙版本": "5.2",
      "电池续航": "30小时"
    }
  },
  {
    "id": "p002",
    "name": "智能手表",
    "price": "¥899",
    ...
  }
]
```

### Excel/CSV格式
Excel文件或CSV文件的第一行应为表头，每行代表一个商品，列名应包括：
- id: 商品ID
- name: 商品名称
- description: 商品描述
- price: 商品价格
- brand: 商品品牌
- category: 商品类别

其他字段如specs（规格）、features（特性）等可以是JSON格式的字符串。
"""
    return guide

# 定义工具列表
tools = [
    {
        "name": "validate_product_input",
        "description": "验证商品信息输入是否规范，可以处理文件或直接文本输入",
        "function": validate_product_input
    },
    {
        "name": "process_product_data",
        "description": "处理商品数据并转换为标准JSON格式",
        "function": process_product_data
    },
    {
        "name": "generate_qa_pairs",
        "description": "基于商品数据生成QA对",
        "function": generate_qa_pairs
    }
]

# 创建工具执行器
tool_executor = ToolExecutor(tools)

# QA助手类
class QAAssistant:
    def __init__(self):
        """初始化QA助手"""
        self.state = SessionState()
        
        # 记录关键路径和系统状态
        logger.info(f"QA助手初始化，当前工作目录: {os.path.abspath(os.getcwd())}")
        
        # 检查关键文件是否存在
        key_files = ["product_data_processor.py", "async_qa_generator.py"]
        for file in key_files:
            if os.path.exists(file):
                file_size = os.path.getsize(file)
                logger.info(f"发现关键文件: {file}, 大小: {file_size} 字节")
            else:
                logger.warning(f"关键文件不存在: {file}")
                
        # 检查常用目录是否存在
        for dir_name in ["qa_agent", "vector_store"]:
            if os.path.exists(dir_name) and os.path.isdir(dir_name):
                logger.info(f"找到目录: {dir_name}")
            
        # 检查模型是否可用
        try:
            model_name = model.model_name if hasattr(model, 'model_name') else str(model)
            logger.info(f"使用的聊天模型: {model_name}")
        except Exception as e:
            logger.warning(f"无法获取模型信息: {str(e)}")

    def initialize(self):
        """初始化会话"""
        # 显示初始化状态
        logger.info("正在初始化QA助手会话...")
        
        # 清理之前可能存在的临时文件
        temp_files = ["products_data.json", "qa_output.json"]
        for file in temp_files:
            if os.path.exists(file):
                try:
                    file_size = os.path.getsize(file)
                    logger.info(f"找到之前的临时文件: {file}, 大小: {file_size} 字节")
                except Exception as e:
                    logger.warning(f"检查临时文件 {file} 时出错: {str(e)}")
        
        # 添加系统消息
        system_message = HumanMessage(content="系统初始化")
        self.state.messages.append(system_message)
        
        # 添加欢迎消息
        welcome_message = """您好！我是电商商品QA对生成助手。我可以帮您将商品信息转换为自然的问答对，模拟电商平台上顾客和客服的交流。

请提供您的商品信息，可以是TXT、DOCX、XLSX、CSV、JSON文件路径，或者直接输入商品数据。我会引导您完成整个生成过程。"""
        
        # 添加额外的文件提示
        try:
            example_files = []
            for example_file in ["example_product.txt", "test_product.txt"]:
                if os.path.exists(example_file):
                    example_files.append(example_file)
            
            if example_files:
                file_list = ", ".join(f"'{f}'" for f in example_files)
                welcome_message += f"\n\n我发现您的系统中有以下示例文件可以使用: {file_list}"
        except Exception as e:
            logger.warning(f"检查示例文件时出错: {str(e)}")
        
        # 添加助手消息
        self.state.messages.append(AIMessage(content=welcome_message))
        return welcome_message
    
    def get_system_prompt(self) -> str:
        """获取系统提示"""
        return f"""你是一个专业的电商商品QA对生成助手。
你负责引导用户完成整个商品QA对生成的流程，包括：
1. 接收并验证商品信息输入
2. 处理商品数据
3. 确认用户意愿后生成QA对

重要规则：
- 你不能自己处理数据，必须使用相应的工具
- 输入验证和数据处理结果由工具返回，你应直接使用这些结果
- 每一步操作前都应向用户确认
- 保持专业友好的语气，使用简洁明了的语言

当前状态：
- 输入源：{self.state.input_source}
- 数据是否已处理：{self.state.data_processed}
- 处理后的数据文件：{self.state.products_file}
- QA对数量：{self.state.qa_count}
- QA生成是否开始：{self.state.qa_generation_started}
- 生成的QA文件：{self.state.qa_file}

请根据当前状态，决定下一步行动。"""
    
    def get_last_assistant_message(self):
        """获取最后一条助手消息"""
        for msg in reversed(self.state.messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return None
    
    def get_last_human_message(self):
        """获取最后一条人类消息"""
        for msg in reversed(self.state.messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None
    
    def process_user_input(self, user_input: str):
        """处理用户输入"""
        # 添加用户消息到历史记录
        self.state.messages.append(HumanMessage(content=user_input))
        
        # 判断当前阶段，执行相应处理
        if self._should_validate_input(user_input):
            return self._handle_input_validation(user_input)
        elif self._should_process_data(user_input):
            return self._handle_data_processing()
        elif self._should_start_qa_generation(user_input):
            return self._handle_qa_generation(user_input)
        else:
            # 默认情况：生成下一步回复
            return self._generate_next_response(user_input)
    
    def _should_validate_input(self, user_input: str) -> bool:
        """判断是否应该验证输入"""
        if self.state.input_source is not None:
            return False
            
        # 判断输入是否可能是商品数据
        return (
            os.path.exists(user_input) or  # 可能是文件路径
            user_input.startswith("{") or  # 可能是JSON
            user_input.startswith("[") or  # 可能是JSON数组
            len(user_input.strip().split("\n")) > 3  # 可能是多行文本
        )
    
    def _should_process_data(self, user_input: str) -> bool:
        """判断是否应该处理数据"""
        if not self.state.input_source or self.state.data_processed:
            return False
            
        # 用户确认继续
        confirmation_patterns = ["是", "确认", "继续", "处理", "下一步", "可以", "好", "同意", "开始"]
        return any(pattern in user_input for pattern in confirmation_patterns)
    
    def _should_start_qa_generation(self, user_input: str) -> bool:
        """判断是否应该开始生成QA对"""
        if not self.state.data_processed or self.state.qa_generation_started:
            return False
            
        # 检查用户输入是否包含数字（可能是QA对数量）
        import re
        number_match = re.search(r'\d+', user_input)
        return number_match is not None
    
    def _handle_input_validation(self, user_input: str):
        """处理输入验证"""
        tool_input = {"input_source": user_input}
        tool_result = tool_executor.execute("validate_product_input", tool_input)
        
        if tool_result["is_valid"]:
            # 更新状态
            self.state.input_source = user_input
            self.state.input_type = tool_result["input_type"]
            
            # 生成回复
            reply = f"我已验证您提供的输入，格式有效。\n\n{tool_result['message']}\n\n您确认继续处理这些数据吗？如果确认，我将把它们转换为标准格式以便生成QA对。"
        else:
            # 生成回复
            reply = f"您提供的输入格式不符合要求：{tool_result['reason']}\n\n请参考以下格式指南调整您的输入:\n\n{tool_result['format_guide']}"
        
        # 添加助手消息
        self.state.messages.append(AIMessage(content=reply))
        return reply
    
    def _handle_data_processing(self):
        """处理数据"""
        tool_input = {
            "input_source": self.state.input_source,
            "input_type": self.state.input_type,
            "output_json": "products_data.json"
        }
        tool_result = tool_executor.execute("process_product_data", tool_input)
        
        if tool_result["success"]:
            # 更新状态
            self.state.data_processed = True
            self.state.products_file = tool_result["output_file"]
            
            # 生成回复
            reply = f"商品数据处理成功！{tool_result['message']}\n\n现在我们可以基于这些数据生成QA对。请告诉我每个商品需要生成多少对QA？"
        else:
            # 生成回复
            reply = f"处理商品数据时出错：{tool_result['reason']}\n\n请检查您的输入并重试。"
        
        # 添加助手消息
        self.state.messages.append(AIMessage(content=reply))
        return reply
    
    def _handle_qa_generation(self, user_input: str):
        """处理QA生成"""
        # 提取数字
        import re
        number_match = re.search(r'\d+', user_input)
        num_pairs = int(number_match.group())
        
        # 更新状态
        self.state.qa_count = num_pairs
        self.state.qa_generation_started = True
        
        # 添加等待消息
        wait_message = f"好的，我将为每个商品生成{num_pairs}对QA。这个过程可能需要一些时间，请稍候..."
        self.state.messages.append(AIMessage(content=wait_message))
        
        # 确保输出文件为绝对路径
        output_file = "qa_output.json"
        output_file_abs = os.path.abspath(output_file)
        logger.info(f"QA输出文件将保存至: {output_file_abs}")
        
        # 检查产品数据文件
        products_file_abs = os.path.abspath(self.state.products_file)
        if not os.path.exists(products_file_abs):
            error_message = f"找不到商品数据文件: {products_file_abs}"
            logger.error(error_message)
            completion_message = f"生成QA对时出错：{error_message}\n\n请确认商品数据是否已成功处理。"
            self.state.messages.append(AIMessage(content=completion_message))
            self.state.qa_generation_started = False
            return completion_message
        
        # 检查产品数据文件大小
        try:
            file_size = os.path.getsize(products_file_abs)
            logger.info(f"商品数据文件大小: {file_size} 字节")
            
            if file_size < 10:  # 文件过小，可能为空
                error_message = f"商品数据文件过小或为空: {file_size} 字节"
                logger.error(error_message)
                completion_message = f"生成QA对时出错：{error_message}\n\n请确认商品数据是否已成功处理。"
                self.state.messages.append(AIMessage(content=completion_message))
                self.state.qa_generation_started = False
                return completion_message
        except Exception as e:
            logger.warning(f"检查商品数据文件大小时出错: {str(e)}")
        
        # 执行生成
        tool_input = {
            "json_file": self.state.products_file,
            "num_pairs": num_pairs,
            "output_file": output_file,
            "concurrency": 3
        }
        
        try:
            # 记录开始时间
            start_time = time.time()
            logger.info(f"开始调用generate_qa_pairs工具，参数: {tool_input}")
            
            # 执行生成
            tool_result = tool_executor.execute("generate_qa_pairs", tool_input)
            
            # 记录完成时间
            end_time = time.time()
            logger.info(f"generate_qa_pairs调用完成，耗时: {end_time - start_time:.2f}秒")
            logger.info(f"工具返回结果: {tool_result}")
            
            if tool_result["success"]:
                # 更新状态
                self.state.qa_file = tool_result["output_file"]
                
                # 检查文件是否实际存在
                output_exists = os.path.exists(output_file)
                logger.info(f"QA输出文件 {output_file} 存在性检查: {'成功' if output_exists else '失败'}")
                
                # 确认文件位置
                abs_path = os.path.abspath(output_file)
                logger.info(f"QA输出文件的绝对路径: {abs_path}")
                
                # 读取示例
                examples = ""
                try:
                    if output_exists:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            qa_pairs = json.load(f)
                            
                        if qa_pairs:
                            # 计算每个商品的QA对数量
                            product_qa_counts = {}
                            for qa in qa_pairs:
                                product_id = qa.get('id', 'unknown')
                                product_qa_counts[product_id] = product_qa_counts.get(product_id, 0) + 1
                            
                            logger.info(f"各商品的QA对数量: {product_qa_counts}")
                            
                            # 显示前3个示例
                            sample_size = min(3, len(qa_pairs))
                            examples = "\n\n示例QA对：\n\n"
                            
                            for i in range(sample_size):
                                examples += f"问题 {i+1}: {qa_pairs[i]['question']}\n"
                                examples += f"回答 {i+1}: {qa_pairs[i]['answer']}\n\n"
                            
                            examples += f"... 等共 {len(qa_pairs)} 对QA已生成"
                    else:
                        examples = "\n\n警告：无法找到QA输出文件，请检查日志以获取详细错误信息。"
                except Exception as e:
                    logger.error(f"读取QA示例时出错: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    examples = f"\n\n读取QA示例时出错: {str(e)}"
                
                # 生成完成消息
                file_path_info = f"所有QA对已保存到文件 {abs_path}。" if output_exists else f"QA文件应当保存在 {abs_path}，但找不到该文件，请检查日志。"
                completion_message = f"{tool_result['message']}！\n\n{file_path_info}{examples}\n\n生成过程已完成。如需要进一步调整或有其他需求，请告诉我。"
                self.state.workflow_completed = True
            else:
                # 生成错误消息
                completion_message = f"生成QA对时出错：{tool_result['reason']}\n\n请检查商品数据并重试。"
                self.state.qa_generation_started = False
            
            # 添加完成消息
            self.state.messages.append(AIMessage(content=completion_message))
            return completion_message
            
        except Exception as e:
            logger.error(f"执行QA生成时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            error_message = f"生成QA对时发生错误: {str(e)}\n\n请检查日志获取详细信息，并重试。"
            self.state.messages.append(AIMessage(content=error_message))
            self.state.qa_generation_started = False
            return error_message
    
    def _generate_next_response(self, user_input: str):
        """生成下一步回复"""
        # 准备提示
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # 准备输入
        chain_input = {
            "history": self.state.messages[:-1],  # 不包括最后一条人类消息
            "input": user_input
        }
        
        # 生成回复
        chain = prompt | model
        response = chain.invoke(chain_input)
        
        # 添加助手消息
        self.state.messages.append(AIMessage(content=response.content))
        return response.content

# 主函数
def main():
    """主函数"""
    # 配置更详细的日志记录
    file_handler = logging.FileHandler("qa_agent.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("\n" + "="*60)
    logger.info("电商商品QA对生成系统启动")
    logger.info("="*60)
    
    print("电商商品QA对生成系统已启动，正在初始化...")
    
    # 检查关键目录和文件
    try:
        # 检查工作目录权限
        cwd = os.getcwd()
        test_file = os.path.join(cwd, ".write_test")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"工作目录 {cwd} 有写入权限")
        except Exception as e:
            logger.warning(f"工作目录 {cwd} 可能没有写入权限: {str(e)}")
        
        # 检查必要文件
        required_files = ["product_data_processor.py", "async_qa_generator.py"]
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"关键文件不存在: {file}")
                print(f"错误: 找不到必要的文件 {file}，请确保其在当前目录中")
                return
        
        # 检查输出目录
        output_dir = "output"
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logger.info(f"已创建输出目录: {output_dir}")
            except Exception as e:
                logger.warning(f"无法创建输出目录: {str(e)}")
    except Exception as e:
        logger.error(f"系统初始化检查时出错: {str(e)}")
    
    try:
        # 创建助手实例
        assistant = QAAssistant()
        
        # 初始化会话
        initial_message = assistant.initialize()
        print(f"\nQA助手: {initial_message}")
        
        # 主循环
        while True:
            try:
                # 获取用户输入
                user_input = input("\n您: ")
                
                # 记录用户输入（但不记录潜在敏感信息）
                safe_input = user_input
                if len(safe_input) > 50:
                    safe_input = safe_input[:50] + "..."
                logger.info(f"用户输入: {safe_input}")
                
                # 检查是否退出
                if user_input.lower() in ["退出", "exit", "quit", "q"]:
                    print("\nQA助手: 感谢使用电商商品QA对生成系统，再见！")
                    logger.info("用户请求退出系统")
                    break
                
                # 处理用户输入
                logger.info("开始处理用户输入...")
                response = assistant.process_user_input(user_input)
                print(f"\nQA助手: {response}")
                logger.info("响应已生成并显示给用户")
                
                # 检查工作流是否已完成
                if assistant.state.workflow_completed:
                    logger.info("工作流已完成")
                    print("\n工作流已完成。您可以继续操作或输入'退出'结束程序。")
            except KeyboardInterrupt:
                print("\n\n操作已取消。您可以继续操作或输入'退出'结束程序。")
                logger.info("用户取消了操作 (KeyboardInterrupt)")
            except Exception as e:
                logger.error(f"处理用户输入时出错: {str(e)}")
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"详细错误: {error_trace}")
                print(f"\nQA助手: 抱歉，处理您的输入时出现了问题: {str(e)}\n请重试或联系管理员。")
    except Exception as e:
        logger.error(f"系统运行时发生严重错误: {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"详细错误: {error_trace}")
        print(f"系统运行时发生错误: {str(e)}\n请查看日志获取详细信息。")
    
    logger.info("电商商品QA对生成系统已关闭")
    print("\n感谢使用电商商品QA对生成系统！")

if __name__ == "__main__":
    main() 