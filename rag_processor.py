from pymilvus import MilvusClient
from pymilvus import model as milvus_model
from pymilvus import connections
import json
import re
import torch
import numpy as np


class RAGProcessor:
    """
    处理 RAG (Retrieval-Augmented Generation) 相关的所有逻辑
    """
    def __init__(self, embedding_dim=1024, chunk_size=512, chunk_overlap=50):
        """
        初始化 RAG 处理器
        :param embedding_dim: 向量嵌入的维度
        :param chunk_size: 文本切片大小
        :param chunk_overlap: 切片重叠大小
        """
        # 设置参数
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化向量数据库
        self.vector_db = self._init_vector_db()
        
        # 初始化默认嵌入模型
        self.embedding_fn = milvus_model.DefaultEmbeddingFunction()

    def _init_vector_db(self):
        """初始化向量数据库连接"""
        # try:
        #     # 首先尝试连接 Milvus 服务器
        #     connections.connect(
        #         alias="default",
        #         host='localhost',
        #         port='19530'
        #     )
        #     print("成功连接到 Milvus 服务器")
        #     return None
        # except Exception as e:
        #     print(f"\n连接 Milvus 服务器失败: {e}")
        #     print("切换到本地 MilvusLite 模式...")
            
        try:
            local_client = MilvusClient("milvus_local.db")
            collection_name = "requirements_collection"
            
            if local_client.has_collection(collection_name=collection_name):
                local_client.drop_collection(collection_name=collection_name)
            
            local_client.create_collection(
                collection_name=collection_name,
                dimension=self.embedding_dim,
            )
            
            print("成功初始化本地 MilvusLite 数据库")
            return local_client
        except Exception as local_e:
            print(f"初始化本地数据库失败: {local_e}")
            return None

    def _chunk_text(self, text):
        """将长文本切分成小块"""
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
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
            start = end - self.chunk_overlap
        
        return chunks

    def _get_embeddings(self, texts):
        """生成文本的向量表示"""
        try:
            vectors = self.embedding_fn.encode_documents(texts)
            return vectors
        except Exception as e:
            print(f"生成嵌入向量时出错: {e}")
            return []

    def store_texts(self, texts, metadata=None):
        """
        存储文本到向量数据库
        :param texts: 文本列表
        :param metadata: 元数据列表
        :return: 是否成功
        """
        try:
            all_chunks = []
            chunk_metadata = []
            
            for i, text in enumerate(texts):
                current_metadata = metadata[i] if metadata and i < len(metadata) else {}
                chunks = self._chunk_text(text)
                
                for j, chunk in enumerate(chunks):
                    chunk_meta = current_metadata.copy()
                    chunk_meta.update({
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'original_text': text,
                        'chunk_text': chunk
                    })
                    chunk_metadata.append(chunk_meta)
                
                all_chunks.extend(chunks)
            
            vectors = self._get_embeddings(all_chunks)
            if not vectors:
                raise Exception("生成向量失败")
            
            data = []
            for i, (chunk, vector, meta) in enumerate(zip(all_chunks, vectors, chunk_metadata)):
                item = {
                    "id": i,
                    "vector": vector,
                    "text": chunk,
                    "metadata": meta
                }
                data.append(item)
            
            if self.vector_db:
                self.vector_db.insert(
                    collection_name="requirements_collection",
                    data=data
                )
                print(f"成功存储 {len(data)} 个文本块到本地数据库")
                return True
            return False
            
        except Exception as e:
            print(f"存储文本时出错: {e}")
            return False

    def search_similar_texts(self, query_text, top_k=5):
        """
        搜索相似文本，同时使用语义检索和关键词检索
        :param query_text: 查询文本
        :param top_k: 返回结果数量
        :return: 相似文本列表
        """
        try:
            # 1. 语义检索
            query_vector = self._get_embeddings([query_text])[0]
            semantic_results = []
            
            if self.vector_db:
                semantic_results = self.vector_db.search(
                    collection_name="requirements_collection",
                    data=[query_vector],
                    limit=top_k,
                    output_fields=["text", "metadata"]
                )
                
                semantic_texts = []
                for hit in semantic_results[0]:
                    semantic_texts.append({
                        'text': hit.entity.get('text'),
                        'metadata': hit.entity.get('metadata'),
                        'score': hit.score,
                        'search_type': 'semantic'
                    })
            else:
                semantic_texts = []

            # 2. 关键词检索
            keyword_results = []
            if self.vector_db:
                # 将查询文本分词
                keywords = query_text.split()
                
                # 对每个关键词进行检索
                for keyword in keywords:
                    if len(keyword) > 1:  # 忽略单字符关键词
                        results = self.vector_db.search(
                            collection_name="requirements_collection",
                            data=[self._get_embeddings([keyword])[0]],
                            limit=top_k,
                            output_fields=["text", "metadata"]
                        )
                        
                        for hit in results[0]:
                            keyword_results.append({
                                'text': hit.entity.get('text'),
                                'metadata': hit.entity.get('metadata'),
                                'score': hit.score * 0.8,  # 降低关键词匹配的权重
                                'search_type': 'keyword',
                                'matched_keyword': keyword
                            })

            # 3. 合并结果
            all_results = semantic_texts + keyword_results
            
            # 4. 去重和排序
            seen_texts = set()
            unique_results = []
            
            for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
                text = result['text']
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_results.append(result)
                    if len(unique_results) >= top_k:
                        break

            return unique_results
                
        except Exception as e:
            print(f"搜索相似文本时出错: {e}")
            return [] 