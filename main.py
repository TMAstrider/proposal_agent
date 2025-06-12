import json
import os
import rules
from dotenv import load_dotenv
from openai import OpenAI
import os

from pymilvus import MilvusClient

from pymilvus import model as milvus_model


class IRARS:
    """
    智能需求分析推荐系统 (Intelligent Requirement Analysis and Recommendation System).
    """
    def __init__(self, knowledge_base_path='knowledge_base.json'):
        """
        初始化系统，加载知识库和规则。
        :param knowledge_base_path: 知识库文件的路径。
        """
        load_dotenv()
        print(f"\napi_key: {os.getenv('SILICONFLOW_API_KEY')}\n")
        self.openai_client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url='https://api.siliconflow.cn/v1',
        )
        self.knowledge_base = self._load_json(knowledge_base_path)
        self.rules = self._load_rules()
        self._init_milvus()


    def _init_milvus(self):
        """初始化 Milvus 连接。"""
        
        self.milvus_client = MilvusClient("milvus_demo.db")

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
        if hasattr(rules, 'RECOMMENDATION_RULES'):
            return rules.RECOMMENDATION_RULES
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
            你是一个专业的软件需求分析师。请分析以下用户需求文本，并从中提取出关键信息。
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
                model="Qwen/QwQ-32B",
                messages=[
                    {"role": "system", "content": "你是一个专业的软件需求分析师，总是以结构化的 JSON 格式输出。请确保输出是有效的 JSON 格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                top_p=0.7,
                frequency_penalty=0.5,
                max_tokens=2048  # 增加 token 限制以确保完整的 JSON 输出
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
            - 文档识别和处理
            - 知识库构建和问答
            - 数据分析和可视化
            - 用户认证和权限管理
            - 系统集成和API对接
            - 实时数据处理
            - 报表生成
            - 工作流自动化

            请仔细分析以下实体信息，并生成最相关的关键词列表：
            {entities_str}

            要求：
            1. 只输出关键词列表，不要包含任何解释或额外文本
            2. 每个关键词应该是标准化的功能名称
            3. 关键词数量控制在3-8个之间
            4. 输出格式必须是有效的 JSON 数组，例如：["关键词1", "关键词2", "关键词3"]
            """

            response = self.openai_client.chat.completions.create(
                model="Qwen/QwQ-32B",
                messages=[
                    {"role": "system", "content": "你是一个专业的产品功能分析师，总是以JSON数组格式输出关键词列表。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                top_p=0.7,
                frequency_penalty=0.5,
                max_tokens=1024
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

    def _match_modules(self, keywords):
        """
        根据关键词匹配知识库中的模块。
        :param keywords: 关键词列表。
        :return: 匹配到的模块列表。
        """
        print(f"用关键词 {keywords} 匹配模块...")
        pass

    def _apply_rules(self, matched_modules):
        """
        应用规则引擎，根据匹配到的模块进行推理。
        :param matched_modules: 匹配到的模块列表。
        :return: 推理结果。
        """
        print(f"向 {matched_modules} 应用规则...")
        pass

    def _generate_output(self, analysis_result):
        """
        根据分析结果生成最终的推荐或输出。
        :param analysis_result: 分析和推理的结果。
        :return: 格式化的输出字符串。
        """
        print(f"为结果 {analysis_result} 生成输出...")
        pass

    def analyze(self, user_input):
        """
        分析用户输入并提供推荐。
        :param user_input: 用户输入的文本。
        :return: 分析和推荐的结果。
        """
        entities = self._extract_entities(user_input)
        keywords = self._generate_keywords(entities)
        matched_modules = self._match_modules(keywords)
        analysis_result = self._apply_rules(matched_modules)
        output = self._generate_output(analysis_result)
        return output

if __name__ == '__main__':
    # 示例用法
    # 请确保您已经在 .env 文件中设置了 SILICONFLOW_API_KEY
    irars_system = IRARS()
    user_query = "我需要开发一个在线商城。核心功能是用户可以浏览商品、加入购物车并在线支付。我们希望这个系统能跑在标准的云服务器上，并且未来能集成第三方的库存管理系统。最终目标是提升销售额和用户满意度。"
    
    print(f"用户需求: {user_query}\n")
    
    # 测试实体提取
    print("\n----------- 实体提取 -----------")
    entities = irars_system._extract_entities(user_query)
    if entities:
        print(json.dumps(entities, indent=2, ensure_ascii=False))
    print("------------------------------------")

    # 测试关键词生成
    print("\n----------- 关键词生成 -----------")
    keywords = irars_system._generate_keywords(entities)
    if keywords:
        print("\n最终生成的关键词列表:")
        print(json.dumps(keywords, indent=2, ensure_ascii=False))
    print("------------------------------------")

    # 完整的分析流程（暂时注释掉，因为其他方法还是空的）
    # print("\n----------- 完整分析流程 -----------")
    # recommendation = irars_system.analyze(user_query)
    # print(f"分析结果: {recommendation}")
    # print("------------------------------------")
