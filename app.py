import streamlit as st
from main import IRARS  
import os

# 初始化智能体
agent = IRARS()

# 页面配置
st.set_page_config(page_title="智能需求分析助手", layout="wide")

# 标题
st.title("智能需求分析助手")

# 输入区域
with st.expander("输入需求描述", expanded=True):
    user_input = st.text_area("请输入您的需求描述:", height=150)
    keywords = st.text_input("关键词(可选):")
    submit = st.button("分析需求")

# 结果展示区域
if submit and user_input:
    with st.spinner("正在分析需求..."):
        # 调用智能体分析
        result = agent.analyze(user_input, keywords)
        
        # 显示结果
        st.subheader("分析结果")
        st.json(result)
        
        # 显示原始召回数据（可选）
        if st.checkbox("显示详细召回数据"):
            st.subheader("召回数据")
            st.write(agent.recall_data)  # 假设您的类中有这个属性

# 侧边栏配置
with st.sidebar:
    st.header("配置")
    model_version = st.selectbox(
        "模型版本",
        ("Qwen3-30B-A3B", "Qwen2-7B")
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    st.info("""
    ### 使用说明
    1. 在输入框中描述您的需求
    2. 可添加关键词辅助分析
    3. 点击"分析需求"按钮获取结果
    """)