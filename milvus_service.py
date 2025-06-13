from pymilvus import MilvusClient
from pymilvus import model as milvus_model
import json
import os
import re

class MilvusService:
    def __init__(self, db_path="milvus_demo.db"):
        """
        初始化 Milvus 服务
        :param db_path: Milvus 数据库文件路径
        """
        self.client = MilvusClient(db_path)
        self.embedding_fn = milvus_model.DefaultEmbeddingFunction()
        self.collection_name = "demo_collection"
        self._init_collection()

    def _init_collection(self):
        """初始化集合"""
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=768,  # 向量维度
        )

    def _chunk_markdown(self, content, chunk_size=512, overlap=50):
        """
        根据 chunk_id 和长度策略对 Markdown 文本进行分块
        :param content: Markdown 文本内容
        :param chunk_size: 块大小
        :param overlap: 重叠大小
        :return: 文本块列表和元数据列表
        """
        print("\n开始处理 Markdown 内容...")
        print(f"内容长度: {len(content)} 字符")
        
        # 使用正则表达式匹配 chunk_id
        chunk_pattern = r'"chunk_id":\s*"([^"]+)"'
        
        # 找到所有 chunk_id 的位置
        chunk_matches = list(re.finditer(chunk_pattern, content))
        print(f"找到 {len(chunk_matches)} 个 chunk_id 标记")
        
        if not chunk_matches:
            print("未找到 chunk_id 标记，将按长度分块")
            chunks = self._chunk_by_length(content, chunk_size, overlap)
            return chunks, [{'is_sub_chunk': False} for _ in chunks]
        
        chunks = []
        metadata_list = []
        
        # 处理每个 chunk_id 之间的内容
        for i in range(len(chunk_matches)):
            start_pos = chunk_matches[i].start()
            chunk_id = chunk_matches[i].group(1)
            print(f"\n处理 chunk_id: {chunk_id}")
            
            # 确定当前块的结束位置
            if i < len(chunk_matches) - 1:
                end_pos = chunk_matches[i + 1].start()
            else:
                end_pos = len(content)
            
            # 提取当前块的内容
            chunk_content = content[start_pos:end_pos].strip()
            print(f"块内容长度: {len(chunk_content)} 字符")
            
            # 如果内容超过 chunk_size，需要进一步分割
            if len(chunk_content) > chunk_size:
                print(f"块内容超过 {chunk_size} 字符，进行子分块")
                sub_chunks = self._chunk_by_length(chunk_content, chunk_size, overlap)
                print(f"分成 {len(sub_chunks)} 个子块")
                for j, sub_chunk in enumerate(sub_chunks):
                    chunks.append(sub_chunk)
                    metadata_list.append({
                        'chunk_id': chunk_id,
                        'sub_chunk_index': j,
                        'total_sub_chunks': len(sub_chunks),
                        'is_sub_chunk': True
                    })
            else:
                print("块内容在长度限制内，保持为单个块")
                chunks.append(chunk_content)
                metadata_list.append({
                    'chunk_id': chunk_id,
                    'is_sub_chunk': False
                })
        
        print(f"\n分块完成，共生成 {len(chunks)} 个块")
        return chunks, metadata_list

    def _chunk_by_length(self, text, chunk_size=512, overlap=50):
        """
        按长度对文本进行分块
        :param text: 输入文本
        :param chunk_size: 块大小
        :param overlap: 重叠大小
        :return: 文本块列表
        """
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                last_punctuation = max(
                    text.rfind('。', start, end),
                    text.rfind('？', start, end),
                    text.rfind('！', start, end),
                    text.rfind('.', start, end),
                    text.rfind('?', start, end),
                    text.rfind('!', start, end)
                )
                
                if last_punctuation > start:
                    end = last_punctuation + 1
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks

    def process_rag_data(self, rag_data_dir="rag_data"):
        """
        处理 rag_data 目录下的所有 Markdown 文件
        :param rag_data_dir: rag_data 目录路径
        :return: 处理结果统计
        """
        print(f"\n开始处理 {rag_data_dir} 目录...")
        if not os.path.exists(rag_data_dir):
            print(f"错误: 目录 {rag_data_dir} 不存在")
            return None

        all_chunks = []
        all_metadata = []
        file_count = 0
        total_chunks = 0

        # 遍历目录下的所有 Markdown 文件
        md_files = [f for f in os.listdir(rag_data_dir) if f.endswith('.md')]
        print(f"找到 {len(md_files)} 个 Markdown 文件")
        
        for filename in md_files:
            file_path = os.path.join(rag_data_dir, filename)
            print(f"\n处理文件: {filename}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"文件大小: {len(content)} 字符")
                    
                    chunks, chunk_metadata = self._chunk_markdown(content)
                    print(f"文件分块完成，生成 {len(chunks)} 个块")
                    
                    # 为每个块添加文件信息
                    for i, (chunk, meta) in enumerate(zip(chunks, chunk_metadata)):
                        meta.update({
                            'filename': filename,
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        })
                        all_chunks.append(chunk)
                        all_metadata.append(meta)
                    
                    total_chunks += len(chunks)
                    file_count += 1
                    print(f"文件 {filename} 处理完成")
                    
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                import traceback
                print("错误详情:")
                print(traceback.format_exc())

        if all_chunks:
            print(f"\n开始生成向量，共 {len(all_chunks)} 个块")
            try:
                # 生成向量并存储
                vectors = self.embedding_fn.encode_documents(all_chunks)
                print("向量生成完成")
                
                data = [
                    {
                        "id": i,
                        "vector": vectors[i],
                        "text": chunk,
                        "metadata": meta
                    }
                    for i, (chunk, meta) in enumerate(zip(all_chunks, all_metadata))
                ]
                
                print("开始插入数据到 Milvus...")
                self.client.insert(collection_name=self.collection_name, data=data)
                print("数据插入完成")
                
                return {
                    "processed_files": file_count,
                    "total_chunks": total_chunks,
                    "success": True
                }
            except Exception as e:
                print(f"生成向量或存储数据时出错: {str(e)}")
                import traceback
                print("错误详情:")
                print(traceback.format_exc())
                return {
                    "processed_files": file_count,
                    "total_chunks": total_chunks,
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "processed_files": 0,
            "total_chunks": 0,
            "success": False,
            "error": "没有找到可处理的块"
        }

    def search_similar(self, query_text, limit=2):
        """
        搜索相似文档
        :param query_text: 查询文本
        :param limit: 返回结果数量
        :return: 搜索结果
        """
        query_vectors = self.embedding_fn.encode_queries([query_text])
        results = self.client.search(
            collection_name=self.collection_name,
            data=query_vectors,
            limit=limit,
            output_fields=["text", "metadata"]
        )
        return results

    def get_collection_stats(self):
        """
        获取集合统计信息
        :return: 集合统计信息
        """
        return self.client.get_collection_stats(self.collection_name) 