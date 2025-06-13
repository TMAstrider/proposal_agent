
# 推荐规则
RECOMMENDATION_RULES = [
    {
        "name": "优先使用IDP规则",
        "condition": "'非结构化文档处理' in features or 'OCR识别' in features",
        "action": "推荐使用IDP智能文档审阅平台",
        "priority": 1
    },
    {
        "name": "企业大脑知识库规则",
        "condition": "'知识库构建' in features or '知识问答' in features",
        "action": "推荐使用企业大脑内部搭建知识库实现知识问答",
        "priority": 2
    },
    {
        "name": "评价分析规则",
        "condition": "'评论分析' in features or '评论分类' in features",
        "action": "推荐使用评价宝产品",
        "priority": 3
    }
]

# 配置信息
SPECS = {
    "企业大脑": [
        {
            "功能用途": "意图理解、闲聊问答、知识问答(RAG)、逻辑推理",
            "主推模型": "Qwen2.5-72B-Instruct",
            "备选模型": ["DeepSeek-R1-Distill-Llama-70B", "DeepSeek-R1-Distill-Qwen-32B"],
            "CPU": "16核(2.6GHz 64位x86架构)",
            "内存": "40G",
            "主推显卡": "NVIDIA A100 80G *4",
            "备选显卡": ["NVIDIA 4090 24G *8", "Ascend 910B 32G *8", "Ascend 910A 32G *8"],
            "存储": "至少500G"
        },
        {
            "功能用途": "数据洞察、数据分析(NL2SQL)",
            "主推模型": "DeepSeek-R1-Distill-Llama-70B",
            "备选模型": "DeepSeek-Coder-33B",
            "CPU": "16核",
            "内存": "40G",
            "主推显卡": "NVIDIA A100 80G *4",
            "备选显卡": "NVIDIA 4090 24G *8",
            "存储": "至少500G"
        },
        {
            "功能用途": "嵌入模型(Embedding)",
            "主推模型": "BCE-Embedding-Base-V1",
            "CPU": "1核",
            "内存": "4G",
            "存储": "至少500G"
        },
        {
            "功能用途": "OCR识别(图片格式PDF等)",
            "主推模型": "Ai能力，OCR识别，票据表单识别",
            "CPU": "16核",
            "内存": "32G",
            "主推显卡": "NVIDIA 4090 24G *1",
            "存储": "至少500G"
        }
    ],
    "Agent": [
        {
            "功能用途": "意图理解、任务拆解、组件生成",
            "主推模型": "TARS-RPA-67B",
            "备选模型": ["TARS-RPA-13B", "TARS-RPA-7B(效果较弱)"],
            "CPU": "16核",
            "内存": "40G",
            "主推显卡": "NVIDIA A100 80G *1",
            "备选显卡": "NVIDIA 4090 24G *4",
            "存储": "至少500G"
        },
        {
            "功能用途": "多模态和页面检测",
            "主推模型": "TARS-VL-7B",
            "CPU": "4核",
            "内存": "10G",
            "主推显卡": "NVIDIA A100 80G *1",
            "最低配置": "NVIDIA 4090 24G *1",
            "存储": "至少500G"
        },
        {
            "功能用途": "知识库检索和匹配",
            "主推模型": "BERT",
            "CPU": "16核",
            "内存": "32G",
            "主推显卡": "NVIDIA A100 80G *1",
            "最低配置": "NVIDIA 4090 24G *1",
            "存储": "至少500G"
        }
    ]
}