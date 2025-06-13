import streamlit as st
from main import IRARS
import os
from dotenv import load_dotenv

# 页面配置
st.set_page_config(
    page_title="智能需求分析助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv('SILICONFLOW_API_KEY', '')
if 'base_url' not in st.session_state:
    st.session_state.base_url = os.getenv('OPENAI_API_BASE', 'https://api.siliconflow.cn/v1')

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # API配置
    st.subheader("API 配置")
    api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="输入您的 API Key"
    )
    base_url = st.text_input(
        "API Base URL",
        value=st.session_state.base_url,
        help="输入 API 的基础 URL"
    )
    
    # 模型配置
    st.subheader("模型配置")
    model_name = st.selectbox(
        "选择模型",
        ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B"],
        index=0,
        help="选择要使用的模型"
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
        help="控制输出的多样性"
    )
    
    # 保存配置
    if st.button("保存配置"):
        st.session_state.api_key = api_key
        st.session_state.base_url = base_url
        st.success("配置已保存！")
    
    # 使用说明
    st.markdown("---")
    st.markdown("""
    ### 📖 使用说明
    1. 首先在侧边栏配置 API 和模型参数
    2. 在主界面输入您的需求描述
    3. 点击"分析需求"按钮获取结果
    4. 可以查看详细的分析过程和推荐结果
    """)

# 主界面
st.title("🤖 智能需求分析助手")
st.markdown("""
这是一个基于大模型的智能需求分析系统，可以帮助您分析需求并提供专业的建议。
""")

# 输入区域
with st.expander("📝 输入需求描述", expanded=True):
    user_input = st.text_area(
        "请输入您的需求描述:",
        height=200,
        placeholder="请详细描述您的需求，包括业务背景、具体功能需求等..."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submit = st.button("🚀 分析需求", type="primary", use_container_width=True)
    with col2:
        clear = st.button("🗑️ 清除", use_container_width=True)

# 结果展示区域
if submit and user_input:
    if not st.session_state.api_key:
        st.error("请先在侧边栏配置 API Key！")
    else:
        try:
            with st.spinner("正在分析需求..."):
                # 初始化智能体
                agent = IRARS(
                    api_key=st.session_state.api_key,
                    base_url=st.session_state.base_url,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # 展示思考过程
                st.markdown("## 🤔 正在分析您的需求")
                
                # 1. 实体抽取
                with st.expander("1. 根据您的问题的实体抽取结果", expanded=True):
                    entities = agent._extract_entities(user_input)
                    st.write(entities)
                
                # 2. 关键词生成
                with st.expander("2. 生成的关键词", expanded=True):
                    keywords = agent._generate_keywords(entities)
                    st.write(keywords)
                
                # 3. 文档召回
                with st.expander("3. 召回的相关文档", expanded=True):
                    docs = agent._get_docs(keywords)
                    st.write(docs)
                
                # 4. 最终分析结果
                st.markdown("## 📊 最终分析结果")
                constructed_result = agent._apply_rules_to_prompt(docs, user_input, keywords)
                result = agent._generate_output(constructed_result)
                
                # 直接显示原始输出
                st.markdown(result)
        
        except Exception as e:
            st.error(f"分析过程中出现错误：{str(e)}")

# 清除按钮功能
if clear:
    st.experimental_rerun()