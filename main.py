import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import prompt_componets
from dotenv import load_dotenv
from openai import OpenAI
from milvus_service import MilvusService

from pymilvus import MilvusClient

from pymilvus import model as milvus_model


class IRARS:
    """
    智能需求分析推荐系统 (Intelligent Requirement Analysis and Recommendation System).
    """
    def __init__(self, api_key=None, base_url=None, model_name="Qwen/Qwen3-30B-A3B", temperature=0.7, top_p=0.7):
        """
        初始化系统，加载知识库和规则。
        :param api_key: OpenAI API密钥
        :param base_url: API基础URL
        :param model_name: 使用的模型名称
        :param temperature: 温度参数
        :param top_p: top_p参数
        """
        load_dotenv()
        
        # 使用传入的API密钥或从环境变量获取
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API密钥未提供，请在初始化时提供api_key参数或在.env文件中设置SILICONFLOW_API_KEY")
        
        # 使用传入的base_url或从环境变量获取
        self.base_url = base_url or os.getenv('OPENAI_API_BASE', 'https://api.siliconflow.cn/v1')
        
        # 初始化OpenAI客户端
        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        # 设置模型参数
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        
        # 加载规则和初始化服务
        self.rules = self._load_rules()
        self.milvus_service = MilvusService()
        self._init_rag_data()

    def _init_rag_data(self):
        """初始化 RAG 数据"""
        print("\n正在处理 RAG 数据...")
        result = self.milvus_service.process_rag_data()
        if result and result['success']:
            print(f"成功处理 {result['processed_files']} 个文件，共 {result['total_chunks']} 个文本块")
        else:
            print("处理 RAG 数据失败")

    def _init_milvus_demo(self):
        """初始化 Milvus 演示数据"""
        # 示例文档
        docs = [
            "猫不是狗.",
            "狗不是猫.",
            "老虎是猫.",
        ]
        
        # 插入文档
        self.milvus_service.insert_documents(docs)
        
        # 测试搜索
        query = "猫是不是狗"
        results = self.milvus_service.search_similar(query)
        print("搜索结果:", results)

    def _init_milvus(self):
        """初始化 Milvus 连接。"""
        
        self.milvus_client = MilvusClient("milvus_main.db")

        if self.milvus_client.has_collection(collection_name="demo_collection"):
            self.milvus_client.drop_collection(collection_name="demo_collection")
        self.milvus_client.create_collection(
            collection_name="demo_collection",
            dimension=768,  # The vectors we will use in this demo has 768 dimensions
        )
        # This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
        embedding_fn = milvus_model.DefaultEmbeddingFunction()

        # Text strings to search from.
        docs = [
            "猫不是狗.",
            "狗不是猫.",
            "老虎是猫.",
        ]

        vectors = embedding_fn.encode_documents(docs)
        # The output vector has 768 dimensions, matching the collection that we just created.
        print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

        # Each entity has id, vector representation, raw text, and a subject label that we use
        # to demo metadata filtering later.
        data = [
            {"id": i, "vector": vectors[i], "text": docs[i]}
            for i in range(len(vectors))
        ]

        print("Data has", len(data), "entities, each with fields: ", data[0].keys())
        print("Vector dim:", len(data[0]["vector"]))
        self.milvus_client.insert(collection_name="demo_collection", data=data)


        query_vectors = embedding_fn.encode_queries(["猫是不是狗"])

        res = self.milvus_client.search(
            collection_name="demo_collection",  # target collection
            data=query_vectors,  # query vectors
            limit=2,  # number of returned entities
            output_fields=["text", "subject"],  # specifies fields to be returned
        )

        print(res)



    def _load_json(self, file_path):
        """从 JSON 文件加载数据。"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 未找到。")
            return {}
        except json.JSONDecodeError:
            print(f"错误: 文件 {file_path} 格式不正确。")
            return {}

    def _load_rules(self):
        """从 rules.py 加载规则。"""
        # 假设 rules.py 中定义了一个名为 RECOMMENDATION_RULES 的变量
        if hasattr(prompt_componets, 'RECOMMENDATION_RULES'):
            return prompt_componets.RECOMMENDATION_RULES
        else:
            print("警告: 在 rules.py 中未找到 RECOMMENDATION_RULES。")
            return []

    def _extract_entities(self, user_input):
        """
        从用户输入中提取关键实体（例如功能、模块名）。
        使用 OpenAI API 将非结构化文本转换为结构化 JSON。
        :param user_input: 用户输入的文本。
        :return: 提取出的实体（JSON 对象）。
        """
        print("正在调用 API 分析需求...")
        try:
            prompt = f"""
            你是一个专业的实体抽取大师。请分析以下用户需求文本，并从中提取出关键信息。
            请将提取的信息结构化为 JSON 对象，包含以下四个维度：
            1. `software_features`: 明确提到的软件功能需求。
            2. `hardware_requirements`: 任何与硬件相关的要求或约束。
            3. `data_sources`: 提到的数据来源或需要集成的外部系统。
            4. `customer_goals`: 客户希望通过这个软件达成的最终业务目标。

            如果某个维度没有信息，请返回一个空列表 `[]`。

            用户需求文本如下：
            ---
            {user_input}
            ---

            请严格按照此 JSON 格式输出，不要添加任何额外的解释或注释。
            输出必须是有效的 JSON 格式，以 {{ 开始，以 }} 结束。
            """

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的软件需求分析师，总是以结构化的 JSON 格式输出。请确保输出是有效的 JSON 格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=0.5,
                max_tokens=8192  # 增加 token 限制以确保完整的 JSON 输出
            )
            
            contents = response.choices[0].message.content
            print("\nAPI 返回的原始内容:")
            print(contents)
            print("\n尝试解析 JSON...")
            
            try:
                return json.loads(contents)
            except json.JSONDecodeError as je:
                print(f"\nJSON 解析错误: {je}")
                print("错误位置:", je.pos)
                print("错误行:", je.lineno)
                print("错误列:", je.colno)
                print("\n尝试清理和修复 JSON 字符串...")
                
                # 尝试清理 JSON 字符串
                cleaned_content = contents.strip()
                # 如果内容不是以 { 开始，找到第一个 {
                start_idx = cleaned_content.find('{')
                if start_idx != -1:
                    cleaned_content = cleaned_content[start_idx:]
                # 如果内容不是以 } 结束，找到最后一个 }
                end_idx = cleaned_content.rfind('}')
                if end_idx != -1:
                    cleaned_content = cleaned_content[:end_idx + 1]
                
                print("\n清理后的内容:")
                print(cleaned_content)
                
                try:
                    return json.loads(cleaned_content)
                except json.JSONDecodeError as je2:
                    print(f"\n清理后仍然无法解析 JSON: {je2}")
                    return {}

        except Exception as e:
            print(f"调用 API 时出错: {e}")
            return {}

    def _generate_keywords(self, entities):
        """
        根据提取的实体生成搜索关键词。
        :param entities: 实体列表（JSON 对象）。
        :return: 关键词列表。
        """
        print("\n正在生成关键词...")
        try:
            # 将实体信息转换为格式化的字符串
            entities_str = json.dumps(entities, ensure_ascii=False, indent=2)
            
            prompt = f"""
            你是一个专业的产品功能分析师。请根据以下提取出的实体信息，生成一组标准化的内部功能关键词。
            这些关键词将用于在我们的知识库中进行匹配。

            我们的产品主要能力包括但不限于：
            意图理解、闲聊问答、知识问答（RAG）、评论分析、评论分类、评论标签化、知识库构建、文档分类、数据流转、关键字段审核、非结构化文档处理、数据整合和处理、嵌入模型、ocr识别、idp智能文档处理、任务拆解、多模态与页面检、组件生成、

            请仔细分析以下实体信息，并生成最相关的关键词列表：
            {entities_str}

            要求：
            1. 只输出关键词列表，不要包含任何解释或额外文本
            2. 每个关键词应该是我们的产品主要能力内容例如：意图理解、闲聊问答、知识问答（RAG）、评论分析、知识库构建、文档分类、数据流转、关键字段审核、非结构化文档处理、数据整合和处理、嵌入模型、ocr识别、idp智能文档处理、任务拆解、多模态与页面检、组件生成等等
            3. 关键词数量控制在3-8个之间
            4. 输出格式必须是有效的 JSON 数组，例如：["关键词1", "关键词2", "关键词3"]
            """

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的产品功能分析师，总是以JSON数组格式输出关键词列表。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=0.5,
                max_tokens=8192
            )
            
            contents = response.choices[0].message.content
            print("\nAPI 返回的原始内容:")
            print(contents)
            print("\n尝试解析关键词列表...")
            
            try:
                keywords = json.loads(contents)
                if isinstance(keywords, list):
                    print(f"\n成功生成 {len(keywords)} 个关键词:")
                    for i, keyword in enumerate(keywords, 1):
                        print(f"{i}. {keyword}")
                    return keywords
                else:
                    print("\n错误：API 返回的不是列表格式")
                    return []
            except json.JSONDecodeError as je:
                print(f"\nJSON 解析错误: {je}")
                # 尝试清理和修复 JSON 字符串
                cleaned_content = contents.strip()
                start_idx = cleaned_content.find('[')
                if start_idx != -1:
                    cleaned_content = cleaned_content[start_idx:]
                end_idx = cleaned_content.rfind(']')
                if end_idx != -1:
                    cleaned_content = cleaned_content[:end_idx + 1]
                
                print("\n清理后的内容:")
                print(cleaned_content)
                
                try:
                    keywords = json.loads(cleaned_content)
                    if isinstance(keywords, list):
                        print(f"\n成功生成 {len(keywords)} 个关键词:")
                        for i, keyword in enumerate(keywords, 1):
                            print(f"{i}. {keyword}")
                        return keywords
                    else:
                        print("\n错误：清理后仍然不是列表格式")
                        return []
                except json.JSONDecodeError as je2:
                    print(f"\n清理后仍然无法解析 JSON: {je2}")
                    return []

        except Exception as e:
            print(f"生成关键词时出错: {e}")
            return []

    def _get_docs(self, keywords):
        """
        根据关键词召回知识库中的文档。
        :param keywords: 关键词列表。
        :return: 匹配到的文档列表。
        """
        print(f"用关键词 {keywords} 匹配文档...")
        
        matched_results = []
        
        # 对每个关键词进行检索
        for keyword in keywords:
            print(f"\n检索关键词: {keyword}")
            try:
                # 使用 Milvus 服务进行语义检索
                results = self.milvus_service.search_similar(keyword, limit=2)
                
                # 提取并打印text内容列表
                text_list = [hit['entity']['text'] for hit in results[0]] if results and len(results) > 0 else []
                print("提取的text内容列表:")
                for i, text in enumerate(text_list, 1):
                    print(f"{i}. {text}")
                
                if results and len(results) > 0:
                    # 处理检索结果
                    for hit in results[0]:
                        result = {
                            'keyword': keyword,
                            'text': hit['entity']['text'],
                            'metadata': hit['entity']['metadata'],
                            'score': hit['distance']
                        }
                        matched_results.append(result)
                    
            except Exception as e:
                print(f"检索关键词 '{keyword}' 时出错: {str(e)}")
                continue
        
        # 按相似度排序并去重
        unique_results = []
        seen_texts = set()
        
        for result in sorted(matched_results, key=lambda x: x['score'], reverse=True):
            text = result['text']
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(result)
        
        print(f"\n共找到 {len(unique_results)} 个唯一匹配")
        print(f"匹配到的唯一文档:{seen_texts}")
        return unique_results

    def _apply_rules_to_prompt(self, retrieved_docs, user_query, keywords=""):
        """
        应用规则引擎，根据匹配到的文档进行推理。
        :param retrieved_docs: 召回的文档列表。
        :param user_query: 用户原始查询。
        :param keywords: 生成的关键词列表。
        :return: 构造的prompt
        """
        # 加载规则和SPECS配置
        rules = self._load_rules()
        specs = prompt_componets.SPECS

        prompt = f"""
        你是一个专业的产品功能分析师，根据用户的产品功能需求，结合知识库中的文档和系统配置，进行功能分析和推荐。
        
        公司已有系统配置信息：
        {json.dumps(specs, indent=2, ensure_ascii=False)}
        
        适用规则：
        {json.dumps(rules, indent=2, ensure_ascii=False)}
        
        知识库文档：
        {retrieved_docs}
        
        用户需求：
        {user_query}
        
        
        按照以下格式输出：
        ### 核心的推荐功能
        ### 推荐功能描述
        ### 推荐功能的硬件配置
        ### 如果有其他的方案可以适当给出一个
        ### 最终结论给出一个配置表格清单（简单）

        请根据以上信息，根据**公司已有系统配置**，进行功能分析和推荐，忠于知识库文档以及配置，并说明推荐理由。
        """

        return prompt

    def _generate_output(self, constructed_result):
        """
        根据构造的prompt结果生成最终的推荐输出。
        :param constructed_result: 构造的prompt结果。
        :return: 格式化的推荐输出。
        """
        print("正在调用API生成最终推荐...")
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的产品推荐专家，请根据分析结果生成结构化推荐。"},
                    {"role": "user", "content": constructed_result}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=0.5,
                max_tokens=8192
            )
            
            contents = response.choices[0].message.content
            # print("\nAPI返回的推荐结果:")
            # print(contents)
            
            # 尝试解析JSON格式的推荐结果
            try:
                return json.loads(contents)
            except json.JSONDecodeError:
                # 如果不是JSON格式，直接返回原始内容
                return contents
                
        except Exception as e:
            print(f"生成推荐时出错: {e}")
            return "生成推荐时发生错误"
        
        return ""

    def analyze(self, user_input):
        """
        分析用户输入并提供推荐。
        :param user_input: 用户输入的文本。
        :return: 分析和推荐的结果。
        """
        entities = self._extract_entities(user_input)
        keywords = self._generate_keywords(entities)
        matched_modules = self._get_docs(keywords)
        constructed_result = self._apply_rules_to_prompt(matched_modules, user_input, keywords)
        output = self._generate_output(constructed_result)
        return output

if __name__ == '__main__':
    # 示例用法
    # 请确保您已经在 .env 文件中设置了 SILICONFLOW_API_KEY
    irars_system = IRARS()
    place_holder = "user query:"
    user_query = f"""
    {place_holder}
    案例场景：母婴电商平台的用户评论智能化管理

    企业背景
    某中型母婴电商平台"BabyCare"主营婴幼儿用品，月均订单量超10万，用户评论日均新增5000条。随着业务增长，平台面临两大问题：

    评论处理低效：人工筛选评论耗时耗力，无法快速定位用户对产品质量、物流速度、客服服务的具体反馈。
    需求洞察模糊：用户对商品的评价分散且主观（如"纸尿裤漏尿""奶瓶刻度不清晰"），难以系统化分析高频问题，导致改进措施滞后。
    """
    output = irars_system.analyze(user_query)
    print(output)
     
    # print(f"用户需求: {user_query}\n")
    
    # # 测试实体提取
    # print("\n----------- 实体提取 -----------")
    # entities = irars_system._extract_entities(user_query)
    # if entities:
    #     print(json.dumps(entities, indent=2, ensure_ascii=False))
    # print("------------------------------------")

    # # 测试关键词生成
    # print("\n----------- 关键词生成 -----------")
    # keywords = irars_system._generate_keywords(entities)
    # if keywords:
    #     print("\n最终生成的关键词列表:")
    #     print(json.dumps(keywords, indent=2, ensure_ascii=False))
    # print("------------------------------------")

    # # 完整的分析流程（暂时注释掉，因为其他方法还是空的）
    # # print("\n----------- 完整分析流程 -----------")
    # # recommendation = irars_system.analyze(user_query)
    # # print(f"分析结果: {recommendation}")
    # # print("------------------------------------")

    # print("\n----------- 文档召回 -----------")
    # docs = irars_system._get_docs(keywords)
    # if docs:
    #     print("\n最终召回到的文档列表:")
    #     print(docs)
    # print("------------------------------------")


    # constructed_result = irars_system._apply_rules_to_prompt(docs, user_query, keywords)
    # print(constructed_result)
    # print(irars_system._generate_output(constructed_result))

    