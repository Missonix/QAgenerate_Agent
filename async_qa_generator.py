from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
import json
import os
import time
import random
import asyncio
from typing import List, Dict, Any, Set
import aiofiles
import logging
import uuid

# 初始化同步和异步模型
model = ChatOpenAI(
    api_key="",
    base_url="",
    model="",
    temperature=0.85,  # 增加温度以提高多样性
)

# 使用一个信号量来限制并发请求数量
# 避免发送太多并发请求导致API限制或资源耗尽
MAX_CONCURRENT_REQUESTS = 5
semaphore = None  # 将在main函数中初始化

logger = logging.getLogger(__name__)

async def load_products(file_path):
    """异步加载商品数据"""
    logger.info(f"从 {file_path} 加载商品数据")
    try:
        # 使用aiofiles异步读取文件
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            products = json.loads(content)
        
        # 构建商品字典，以ID为键
        products_dict = {}
        
        # 处理和标准化ID
        for product in products:
            # 优先使用商品ID字段(中文字段名)，这样与QA回答时使用的ID一致
            product_id = product.get('商品ID')
            
            # 如果没有商品ID字段，则使用id字段(英文字段名)
            if not product_id:
                product_id = product.get('id')
            
            # 如果两个字段都没有，生成一个随机ID
            if not product_id:
                product_id = f"gen_{uuid.uuid4().hex[:8]}"
                product['id'] = product_id
                logger.warning(f"商品没有ID，已分配随机ID: {product_id}")
            
            # 确保product同时包含中文ID和英文ID
            if 'id' not in product and '商品ID' in product:
                product['id'] = product['商品ID']
            if '商品ID' not in product and 'id' in product:
                product['商品ID'] = product['id']
            
            # 存储商品
            products_dict[product_id] = product
        
        logger.info(f"成功加载 {len(products_dict)} 个商品")
        return products_dict
    except Exception as e:
        logger.error(f"加载商品数据时出错: {str(e)}")
        raise

def format_product_info(product):
    """格式化产品信息为文本"""
    info = []
    
    # 标准字段
    std_fields = {
        'id': 'ID', 
        '商品ID': '商品ID',
        'name': '名称',
        '商品名称': '商品名称',
        'description': '描述', 
        '商品描述': '商品描述',
        'price': '价格', 
        '价格': '价格',
        'brand': '品牌', 
        '品牌': '品牌',
        'category': '类别',
        '类别': '类别'
    }
    
    # 添加基本信息
    for field, label in std_fields.items():
        if field in product and product[field]:
            info.append(f"{label}: {product[field]}")
    
    # 处理规格或其他嵌套字段
    for field in product:
        if field not in std_fields and isinstance(product[field], dict):
            info.append(f"{field}:")
            for key, value in product[field].items():
                info.append(f"  - {key}: {value}")
        elif field not in std_fields and isinstance(product[field], list):
            info.append(f"{field}:")
            for item in product[field]:
                if isinstance(item, str):
                    info.append(f"  - {item}")
                elif isinstance(item, dict):
                    for key, value in item.items():
                        info.append(f"  - {key}: {value}")
    
    return "\n".join(info)

# 定义不同的关注点，引导模型生成多样化问题
FOCUS_POINTS = [
    "价格和优惠：关注商品的价格、折扣、优惠活动、性价比等",
    "功能特性：关注商品的主要功能、特色功能、创新点等",
    "规格参数：关注商品的技术参数、尺寸、重量、容量等具体数据",
    "使用场景：关注商品适合在什么环境或情况下使用",
    "适用人群：关注商品适合哪类人群使用，如年龄段、职业、兴趣爱好等",
    "与竞品比较：关注商品与其他同类产品的区别、优势",
    "售后服务：关注保修、退换货政策、售后支持等",
    "使用寿命：关注商品的耐用度、电池续航、使用年限等",
    "外观设计：关注商品的颜色、材质、外观设计、时尚度等",
    "使用体验：关注商品的易用性、舒适度、用户评价等",
    "安装调试：关注商品的安装难度、调试方法、兼容性等",
    "配件耗材：关注商品的配件、耗材价格、更换频率等",
    "物流发货：关注发货时间、物流方式、是否包邮等",
    "商品真伪：关注正品保障、防伪验证等",
    "商品库存：关注是否有货、什么时候能发货等"
]

# 约束列表，用于增加问题多样性
CONSTRAINTS = [
    "问题要非常简短，不超过10个字",
    "问题要包含一个疑问词，如'吗'、'呢'、'吧'等",
    "问题要直接了当，不需要礼貌用语",
    "问题要像在跟朋友聊天一样随意",
    "问题可以使用一些网络用语或缩写",
    "问题可以包含一些表情符号",
    "问题要像匆忙打字一样，可以省略一些字"
]

# 使用线程安全的集合跟踪已使用的关注点和问题
class SafeSet:
    def __init__(self):
        self.data = set()
        self.lock = asyncio.Lock()
    
    async def add(self, item):
        async with self.lock:
            self.data.add(item)
    
    async def contains(self, item):
        async with self.lock:
            return item in self.data
    
    async def clear(self):
        async with self.lock:
            self.data.clear()
    
    async def get_copy(self):
        async with self.lock:
            return self.data.copy()

# 创建用于跟踪已使用关注点和问题的安全集合
used_focuses = SafeSet()
used_questions = SafeSet()

async def generate_question(product_info, product_name, qa_id):
    """异步生成问题"""
    # 使用信号量限制并发请求
    async with semaphore:
        # 随机选择一个未使用的关注点
        all_focuses = FOCUS_POINTS.copy()
        used = await used_focuses.get_copy()
        available_focuses = [f for f in all_focuses if f not in used]
        
        if not available_focuses:  # 如果所有关注点都已使用，则重置
            await used_focuses.clear()
            available_focuses = all_focuses
        
        focus = random.choice(available_focuses)
        await used_focuses.add(focus)
        
        # 尝试三次
        for attempt in range(3):
            try:
                # 添加随机约束以增加多样性
                random_constraint = random.choice(CONSTRAINTS)
                
                question_prompt = f"""你是一个正在电商平台购物的顾客，请根据以下商品信息，生成一个真实自然的问题。

要求：
1. 问题要简短直接，像真实顾客一样提问
2. 不要使用复杂的句式，要口语化
3. 不要过度礼貌或正式，要像日常聊天一样随意
4. 【特别要求】{random_constraint}
5. 【重要】请特别关注商品的这个方面：{focus}

以下是一些例子：
- 商品: 电吹风
  问题: "这个风力大吗？"
- 商品: 笔记本电脑
  问题: "续航能撑多久？"
- 商品: 运动鞋
  问题: "44码的有吗"
- 商品: 婴儿奶粉
  问题: "保质期多久啊"
- 商品: 洗面奶
  问题: "油皮能用不"
- 商品: 手机
  问题: "支持5G不？"

商品信息:
{product_info}

只生成一个问题，不要有任何其他内容，不要解释，不要引号。记住，要围绕"{focus}"来提问。
"""
                print(f"[{qa_id}] 当前关注点: {focus}")
                question_response = model.invoke(question_prompt)
                question = question_response.content.strip()
                
                # 确保生成的问题不为空，并且与之前的问题不同
                if question and len(question) > 2 and not await used_questions.contains(question):
                    await used_questions.add(question)
                    return question
                else:
                    reason = '太短或为空' if not question or len(question) <= 2 else '与之前的问题重复'
                    print(f"[{qa_id}] 生成的问题{reason}，重试...")
            except Exception as e:
                print(f"[{qa_id}] 生成问题尝试 {attempt+1} 失败: {e}")
                await asyncio.sleep(1)  # 等待一秒再试
        
        # 如果全部尝试都失败，返回一个包含随机关注点的默认问题
        focus_keywords = focus.split("：")[0]
        default_question = f"这款{product_name}的{focus_keywords}怎么样？"
        await used_questions.add(default_question)  # 记录默认问题，避免重复
        print(f"[{qa_id}] 所有问题生成尝试都失败，使用默认问题")
        return default_question

async def generate_answer(product_info, question, qa_id):
    """异步生成回答"""
    # 使用信号量限制并发请求
    async with semaphore:
        # 尝试三次
        for attempt in range(3):
            try:
                answer_prompt = f"""你是一个专业的电商客服代表，请根据提供的商品信息回答顾客的问题。

要求：
1. 回答要像真实电商客服一样，热情有礼貌
2. 一定要用"亲"作为开头称呼客户
3. 句尾经常使用"呢"、"哦"、"呀"等语气词
4. 适当使用emoji表情符号增加亲和力
5. 回答要完整、准确，但语气要轻松活泼

以下是一些例子：
- 问题: "这个风力大吗？"
  回答: "亲～这款电吹风的风力非常强劲呢💨，有三档可调节，最大档位可达1800W功率，能够快速吹干长发哦😊"
  
- 问题: "续航能撑多久？"
  回答: "亲，这款笔记本电脑满电情况下正常使用可续航8-10小时呢🔋，办公模式下甚至能达到12小时哦，是出差旅行的好伙伴呢✨"
  
- 问题: "44码的有吗"
  回答: "亲，这款运动鞋目前44码有库存的呢👟，需要我帮您预订吗？发货很快的哦～😉"
  
- 问题: "保质期多久啊"
  回答: "亲，这款奶粉的保质期是18个月呢🍼，我们发货都是新鲜货源，至少还有一年以上的保质期，请您放心购买哦💕"

商品信息:
{product_info}

顾客问题: {question}

请只给出回答内容，不要添加对话标签或其他格式。如果商品信息中没有相关内容，请明确表示'亲，抱歉，目前没有该商品的相关信息呢😊'。
"""
                answer_response = model.invoke(answer_prompt)
                return answer_response.content.strip()
            except Exception as e:
                print(f"[{qa_id}] 生成回答尝试 {attempt+1} 失败: {e}")
                await asyncio.sleep(1)  # 等待一秒再试
        
        # 如果全部尝试都失败，返回一个默认回答
        print(f"[{qa_id}] 所有回答生成尝试都失败，使用默认回答")
        return "亲，这个问题的答案可以在商品描述中找到呢😊 如果您有其他疑问，可以随时问我哦～"

async def generate_qa_pair(product_id, product, num_pairs=1, start_qa_id=0, product_index=0, total_products=0):
    """异步生成问答对"""
    product_info = format_product_info(product)
    product_name = product.get('name', '')
    tasks = []
    
    print(f"\n开始为商品 [{product_id}]{product_name} 并行生成 {num_pairs} 对QA...")
    
    # 创建多个并行任务
    for i in range(num_pairs):
        qa_id = start_qa_id + i
        # 为每对QA创建一个任务
        tasks.append(generate_single_qa_pair(product_id, product_info, product_name, qa_id, i, num_pairs))
    
    # 并行执行所有任务
    qa_pairs = await asyncio.gather(*tasks)
    
    print(f"\n已完成商品 [{product_id}]{product_name} 的 {len(qa_pairs)} 对QA生成")
    
    return qa_pairs

async def generate_single_qa_pair(product_id, product_info, product_name, qa_id, qa_index, total_qa_for_product):
    """生成单个QA对"""
    print(f"[{qa_id}] 正在为商品 [{product_id}] 生成第 {qa_index+1}/{total_qa_for_product} 对QA...")
    
    # 生成问题
    question = await generate_question(product_info, product_name, qa_id)
    
    # 生成回答
    answer = await generate_answer(product_info, question, qa_id)
    
    print(f"[{qa_id}] 完成商品 [{product_id}] 的第 {qa_index+1}/{total_qa_for_product} 对QA生成")
    
    return {
        "id": product_id,
        "question": question,
        "answer": answer
    }

async def save_qa_pairs(qa_pairs, output_file):
    """异步保存QA对到文件"""
    async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(qa_pairs, ensure_ascii=False, indent=2))
    print(f"已保存 {len(qa_pairs)} 对QA到 {output_file}")

async def process_product(product_id, product, num_pairs, total_qa_pairs, output_file, start_qa_id, product_index=0, total_products=0):
    """处理单个商品，生成QA对并保存"""
    product_name = product.get('name', product.get('商品名称', ''))
    print(f"\n===== 开始处理第 {product_index+1}/{total_products} 个商品 =====")
    print(f"商品ID: {product_id}")
    print(f"商品名称: {product_name}")
    print(f"将为该商品生成 {num_pairs} 对QA")
    print(f"当前已生成总QA数: {len(total_qa_pairs)}")
    
    # 确保我们使用有效的ID，这将是最终保存到QA对中的ID
    # 确保与answer_node函数中使用的ID格式一致
    qa_product_id = product.get('id', product_id)
    
    # 生成该商品的QA对
    product_qa_pairs = await generate_qa_pair(qa_product_id, product, num_pairs, start_qa_id, product_index, total_products)
    
    # 将新生成的QA对添加到总列表中
    total_qa_pairs.extend(product_qa_pairs)
    
    # 保存当前所有QA对
    await save_qa_pairs(total_qa_pairs, output_file)
    
    print(f"\n===== 完成第 {product_index+1}/{total_products} 个商品的处理 =====")
    print(f"为商品 [{qa_product_id}]{product_name} 生成了 {len(product_qa_pairs)} 对QA")
    print(f"当前总QA数: {len(total_qa_pairs)}")
    
    return len(product_qa_pairs)

async def main_async(data_path, product_ids=None, num_pairs=1, output_file="async_qa_output.json", concurrency=3):
    """异步主函数"""
    global semaphore
    semaphore = asyncio.Semaphore(concurrency)
    
    start_time = time.time()
    
    # 加载商品数据
    products = await load_products(data_path)
    
    # 如果未指定商品ID，则使用所有商品
    if product_ids is None:
        product_ids = list(products.keys())
    
    total_products = len(product_ids)
    print(f"\n========== QA生成任务开始 ==========")
    print(f"将为 {total_products} 个商品生成QA对，每个商品 {num_pairs} 对，预计总共 {total_products * num_pairs} 对")
    print(f"商品ID列表: {product_ids}")
    print(f"并发数: {concurrency}")
    
    # 初始化全局列表存储所有QA对
    all_qa_pairs = []
    
    # 在每次运行开始时清空历史记录
    await used_focuses.clear()
    await used_questions.clear()
    
    # 按序处理每个商品（但每个商品内的QA对生成是并行的）
    total_qa_count = 0
    for index, product_id in enumerate(product_ids):
        if product_id in products:
            # 为每个商品生成QA对
            count = await process_product(
                product_id, 
                products[product_id], 
                num_pairs, 
                all_qa_pairs, 
                output_file,
                total_qa_count,
                index,
                total_products
            )
            total_qa_count += count
        else:
            print(f"\n⚠️ 警告: 商品ID [{product_id}] 在数据中不存在，已跳过")
    
    end_time = time.time()
    print(f"\n========== QA生成任务完成 ==========")
    print(f"共生成 {len(all_qa_pairs)} 对QA，耗时 {end_time - start_time:.2f} 秒")
    print(f"平均每对QA生成时间: {(end_time - start_time) / max(1, len(all_qa_pairs)):.2f} 秒")
    print(f"结果已保存至: {output_file}")
    
    return all_qa_pairs

def main(data_path, product_ids=None, num_pairs=1, output_file="async_qa_output.json", concurrency=3):
    """同步主函数入口"""
    return asyncio.run(main_async(data_path, product_ids, num_pairs, output_file, concurrency))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='异步并行生成多样化的电商QA对')
    parser.add_argument('--data_path', type=str, default='products_data.json', help='商品数据文件路径')
    parser.add_argument('--product_ids', type=str, nargs='+', help='要生成QA对的商品ID列表')
    parser.add_argument('--num_pairs', type=int, default=1, help='每个商品生成的QA对数量')
    parser.add_argument('--output', type=str, default='async_qa_output.json', help='输出文件路径')
    parser.add_argument('--concurrency', type=int, default=3, help='并发请求数量（默认3，建议不要太高以避免API限制）')
    
    args = parser.parse_args()
    
    main(args.data_path, args.product_ids, args.num_pairs, args.output, args.concurrency) 