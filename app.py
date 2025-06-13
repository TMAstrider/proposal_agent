import streamlit as st
from main import IRARS
import time

# put your api key here
YOUR_API_KEY = ""

# 设置页面配置
st.set_page_config(
    page_title="Proposal agent for HIAS assignment",
    page_icon="🤖",
    layout="wide"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "base_url" not in st.session_state:
    st.session_state.base_url = ""

# 侧边栏配置
with st.sidebar:
    st.title("⚙️ 配置")
    
    # API Key 配置
    api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key or YOUR_API_KEY,
        type="password",
        help="请输入您的 API Key"
    )
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
    
    # Base URL 配置
    base_url = st.text_input(
        "Base URL",
        value=st.session_state.base_url or "https://api.siliconflow.cn/v1",
        help="请输入 API 基础 URL"
    )
    if base_url != st.session_state.base_url:
        st.session_state.base_url = base_url
    
    # 模型参数配置
    st.markdown("### 模型参数")
    model_name = st.selectbox(
        "模型",
        ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-14B-A3B", "Qwen/Qwen3-7B-A3B"],
        index=0
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="控制输出的随机性，值越大输出越随机"
    )
    
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="控制输出的多样性，值越大输出越多样"
    )

# 主界面
st.title("🤖 智能需求分析助手")

# 创建主容器
main_container = st.container()

# 创建底部输入容器
input_container = st.container()

# 在主容器中显示聊天历史
with main_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 在底部容器中添加输入框
with input_container:
    st.markdown("---")
    if prompt := st.chat_input("请输入您的需求..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 检查 API Key
        if not st.session_state.api_key:
            with st.chat_message("assistant"):
                st.error("请先在侧边栏配置 API Key！")
            st.session_state.messages.append({"role": "assistant", "content": "请先在侧边栏配置 API Key！"})
        else:
            try:
                
                # 初始化智能体
                agent = IRARS(
                    api_key=st.session_state.api_key,
                    base_url=st.session_state.base_url,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # 创建助手消息容器
                with st.chat_message("assistant"):
                    # 创建tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["实体抽取", "关键词生成", "文档召回", "最终分析"])
                    
                    # 1. 实体抽取
                    with tab1:
                        st.markdown("🤔 正在分析实体...")
                        time.sleep(1)
                        entities = agent._extract_entities(prompt)
                        st.write(entities)
                    
                    # 2. 关键词生成
                    with tab2:
                        st.markdown("🔍 正在生成关键词...")
                        time.sleep(1)
                        keywords = agent._generate_keywords(entities)
                        st.write(keywords)
                    
                    # 3. 文档召回
                    with tab3:
                        st.markdown("📚 正在召回相关文档...")
                        time.sleep(1)
                        docs = agent._get_docs(keywords)
                        st.write(docs)
                    
                    # 4. 最终分析结果
                    with tab4:
                        st.markdown("📊 正在生成最终分析...")
                        time.sleep(1)
                        constructed_result = agent._apply_rules_to_prompt(docs, prompt, keywords)
                        result = agent._generate_output(constructed_result)
                        st.markdown(result)
                    
                    # 构建完整响应
                    full_response = f"### 分析结果\n\n**实体抽取结果：**\n```json\n{entities}\n```\n\n**生成的关键词：**\n```json\n{keywords}\n```\n\n**召回的相关文档：**\n```json\n{docs}\n```\n\n**最终分析结果：**\n{result}"
                
                # 添加助手消息到历史
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"分析过程中出现错误：{str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"分析过程中出现错误：{str(e)}"})