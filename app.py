import os
import io
import re
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from anthropic import Anthropic
from openai import OpenAI

# ══════════════════════════════════════════════════════════════
# 页面配置
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="金融风控智能分析助手",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# 厂商 & 模型定义
# ══════════════════════════════════════════════════════════════
PROVIDERS = {
    "Anthropic (Claude)": {
        "models": ["claude-sonnet-5", "claude-haiku-4-5"],
        "env_key": "ANTHROPIC_API_KEY",
    },
    "OpenAI (GPT)": {
        "models": ["gpt-4o", "gpt-4.1-mini"],
        "env_key": "OPENAI_API_KEY",
    },
    "DeepSeek (国产)": {
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "env_key": "DEEPSEEK_API_KEY",
    },
}

# ══════════════════════════════════════════════════════════════
# 系统提示词
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """你是金融风控分析专家，擅长以下领域：
- Vintage分析（账龄分析、迁徙率分析、滚动率分析）
- 归因分析（风险因子拆解、贡献度量化、边际效应分析）
- 三方数据源评估（覆盖率、命中率、区分度、KS/Lift/AUC评估）
- 评分卡建模（WOE/IV、变量分箱、模型稳定性PSI）
- 风控策略监控（通过率、拒绝率、拒绝原因分布、策略触碰分析）
- 冠军挑战者分析（A/B测试、策略效果对比、显著性检验）
- Swap分析（数据源置换评估、增量贡献度测算）

回答要求：
1. 分析结论必须附带数据依据（引用指标数值、对比基准、变化幅度）
2. 涉及预测或判断性结论时，必须附加风险提示：
   「⚠️ AI分析仅供参考，不替代专业风控决策」
3. 使用中文回复，结构清晰，善用标题、表格和小点
4. 如果用户问题超出风控范畴，友好引导回风控主题
5. 如果用户上传了数据文件，优先基于数据进行分析"""

# ══════════════════════════════════════════════════════════════
# 风控分析模板
# ══════════════════════════════════════════════════════════════
TEMPLATES = {
    # ── 风险监控类 ──
    "📊 Vintage账龄分析": (
        "请基于上传数据，进行Vintage账龄分析：\n"
        "1. 计算各放款月度的M0→M1、M1→M2、M2→M3+迁徙率\n"
        "2. 识别迁徙率异常升高的放款批次（高于历史均值1.5倍标准差）\n"
        "3. 分析可能的风险因子（客群变化、政策调整、宏观因素）\n"
        "4. 给出各月度的风险预警阈值建议"
    ),
    "📈 迁徙率滚动分析": (
        "请基于上传数据，进行迁徙率滚动分析：\n"
        "1. 计算各月份的状态流转矩阵（Current→M1、M1→M2、M2→M3、M3→M4+）\n"
        "2. 绘制各月份迁徙率趋势对比\n"
        "3. 按关键维度（渠道、产品、客群分层）拆解迁徙率差异\n"
        "4. 识别迁徙率拐点月份及可能的触发原因"
    ),
    "📋 通过率与拒绝率监控": (
        "请基于上传数据，进行策略通过率与拒绝率监控分析：\n"
        "1. 整体通过率、拒绝率日/周/月度趋势\n"
        "2. 各策略节点的触碰率与拒绝率（按规则/模型/策略层拆解）\n"
        "3. 拒绝原因分布（Top N 拒绝规则/拒绝原因编码）\n"
        "4. 通过率异常波动检测（环比/同比异常超过±3%的时点）\n"
        "5. 不同客群（渠道、产品、地区）的通过率差异分析"
    ),
    "🚫 拒绝原因深度分析": (
        "请基于上传数据，进行拒绝原因深度分析：\n"
        "1. 拒绝原因分布与占比（按一级/二级原因分类）\n"
        "2. 各拒绝原因的逾期表现回溯（被拒客群如果放款，预期坏账率）\n"
        "3. 拒绝规则命中重叠度分析（多规则同时命中的客群占比）\n"
        "4. 识别可能「误拒」的规则（命中率高但对应坏账率低的规则）\n"
        "5. 拒绝规则优化建议（阈值调整、豁免逻辑、灰度方案）"
    ),
    # ── 模型与变量类 ──
    "📐 WOE/IV分箱分析": (
        "请基于上传数据，进行变量分箱与WOE/IV分析：\n"
        "1. 对关键变量进行最优分箱（等频/等距/卡方/决策树分箱对比）\n"
        "2. 计算各分箱的WOE值、IV值\n"
        "3. 变量单调性检验（WOE是否随分箱单调递减/递增）\n"
        "4. 变量区分度评级（IV<0.02 无预测力 / 0.02-0.1 弱 / 0.1-0.3 中等 / >0.3 强）\n"
        "5. 建议入模变量及分箱方案"
    ),
    "🔬 特征变量评估": (
        "请基于上传数据，进行特征变量全面评估：\n"
        "1. 各变量的IV值、KS值、AUC值对比排名\n"
        "2. PSI稳定性指标（按时间/样本对比，PSI>0.25视为不稳定）\n"
        "3. 变量缺失率与异常值分析\n"
        "4. 变量间相关性矩阵（VIF多重共线性检验）\n"
        "5. 变量重要性排序（信息增益/随机森林特征重要性）\n"
        "6. 推荐核心变量组合及淘汰建议"
    ),
    "🎯 评分卡效果评估": (
        "请进行评分卡效果评估：\n"
        "1. 各分数段的好/坏客户分布及Odds值\n"
        "2. 整体KS值与AUC值\n"
        "3. 按时间维度的PSI稳定性（训练集 vs 验证集 vs 近期样本）\n"
        "4. 不同cutoff阈值下的通过率与坏账率权衡\n"
        "5. 评分卡排序能力验证（分数越高坏账率是否单调递减）"
    ),
    # ── 策略决策类 ──
    "🔄 Swap置换分析": (
        "请基于上传数据，进行Swap置换分析：\n"
        "1. 当前数据源A vs 候选数据源B 的覆盖率与命中率对比\n"
        "2. 两数据源的KS值、Lift曲线、AUC差异\n"
        "3. 置换后的增量贡献（叠加入现有策略后的KS提升量）\n"
        "4. 两数据源的重叠客群分析（交集/差集覆盖的客群特征）\n"
        "5. 成本-效益对比：每单位KS提升的成本估算\n"
        "6. 建议：维持现状 / 置换 / 双源并行方案"
    ),
    "🏆 冠军挑战者分析": (
        "请基于上传数据，进行冠军挑战者分析：\n"
        "1. 冠军策略 vs 挑战者策略的通过率、坏账率对比\n"
        "2. 两策略的客群重叠度（一致通过/一致拒绝/分歧客群占比）\n"
        "3. 分歧客群的最终逾期表现（作为胜负判定依据）\n"
        "4. 统计显著性检验（如果样本量充足，进行Z检验或卡方检验）\n"
        "5. 按客群细分（渠道/产品/分数段）的策略效果差异\n"
        "6. 结论：挑战者是否可替换冠军，或建议灰度比例"
    ),
    "📋 三方数据源评估": (
        "请对三方数据源进行评估分析：\n"
        "1. 数据源覆盖率（总体/分客群）与命中率\n"
        "2. KS值、Lift曲线、AUC等区分度指标\n"
        "3. 增量贡献度（在当前模型基础上叠加后的KS/AUC提升）\n"
        "4. 按客群维度（渠道、产品、分数段）的效果差异\n"
        "5. 接入成本与预期收益评估\n"
        "6. 上线后的监控指标体系建议"
    ),
    "🔍 归因分析": (
        "请进行风险归因分析：\n"
        "1. 识别导致逾期率变化的主要影响因子\n"
        "2. 量化各因子的贡献度（边际贡献/Shapley值分解法）\n"
        "3. 区分结构性变化 vs 周期性波动 vs 一次性事件\n"
        "4. 不同客群维度的归因拆解（渠道×产品×时间）\n"
        "5. 提出针对性的风险缓释建议与优先级排序"
    ),
    "💡 自由提问": "",
}

# ══════════════════════════════════════════════════════════════
# 会话状态初始化
# ══════════════════════════════════════════════════════════════
DEFAULTS = {
    "messages": [],
    "uploaded_data": None,
    "uploaded_filename": None,
    "uploaded_columns": [],
    "pending_template": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def build_data_context(df: pd.DataFrame, filename: str) -> str:
    """将上传数据摘要注入上下文"""
    buf = io.StringIO()
    df.head(20).to_string(buf, index=False)
    preview = buf.getvalue()

    stats_buf = io.StringIO()
    df.describe(include="all").to_string(stats_buf)
    stats = stats_buf.getvalue()

    return (
        f"[用户已上传数据文件: {filename}]\n"
        f"行数: {len(df)}, 列数: {len(df.columns)}\n"
        f"字段列表: {list(df.columns)}\n"
        f"数据类型:\n{df.dtypes.to_string()}\n"
        f"数据预览(前20行):\n{preview}\n"
        f"统计摘要:\n{stats[:3000]}\n"
    )


def call_anthropic(api_key: str, model: str, system: str, messages: list, temperature: float) -> str:
    """调用 Anthropic Claude API"""
    client = Anthropic(api_key=api_key)
    conversation = [{"role": m["role"], "content": m["content"]} for m in messages]
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=conversation,
        temperature=temperature,
    )
    return resp.content[0].text


def call_openai_compatible(api_key: str, base_url: str, model: str, system: str, messages: list, temperature: float) -> str:
    """调用 OpenAI / DeepSeek 等兼容 API"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    api_messages = [{"role": "system", "content": system}]
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})
    resp = client.chat.completions.create(
        model=model,
        messages=api_messages,
        max_tokens=4096,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def route_api_call(provider: str, model: str, api_key: str, messages: list, temperature: float) -> str:
    """根据厂商路由到对应 API"""
    if provider.startswith("Anthropic"):
        return call_anthropic(api_key, model, SYSTEM_PROMPT, messages, temperature)
    elif provider.startswith("OpenAI"):
        return call_openai_compatible(api_key, "https://api.openai.com/v1", model, SYSTEM_PROMPT, messages, temperature)
    elif provider.startswith("DeepSeek"):
        return call_openai_compatible(api_key, "https://api.deepseek.com", model, SYSTEM_PROMPT, messages, temperature)
    else:
        raise ValueError(f"不支持的厂商: {provider}")


def export_conversation(messages: list, fmt: str) -> bytes:
    """导出对话记录为指定格式"""
    records = []
    for i, m in enumerate(messages, 1):
        role = "用户" if m["role"] == "user" else "AI分析"
        content = m["content"]
        records.append({"序号": i, "角色": role, "内容": content, "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if i == len(messages) else ""})

    df = pd.DataFrame(records)

    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8-sig")
    elif fmt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="对话记录")
        return buf.getvalue()
    else:  # markdown
        lines = [
            "# 🛡️ 金融风控智能分析助手 - 对话记录",
            f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ]
        for i, m in enumerate(messages, 1):
            role_label = "👤 用户" if m["role"] == "user" else "🤖 AI分析"
            lines.append(f"### {i}. {role_label}")
            lines.append("")
            lines.append(m["content"])
            lines.append("")
            lines.append("---")
            lines.append("")
        lines.append("*⚠️ AI分析仅供参考，不替代专业风控决策*")
        return "\n".join(lines).encode("utf-8")

# ══════════════════════════════════════════════════════════════
# 侧边栏
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── 标题 ──
    st.title("🛡️ 金融风控智能分析助手")
    st.caption(
        "上传风控数据，即刻获得：Vintage迁徙分析 · 策略监控诊断 · "
        "特征变量评估 · 归因与置换分析 · 冠军挑战者对比 · "
        "多模型交叉验证 —— 一站式风控策略分析工作站"
    )
    st.divider()

    # ── 1. 风控分析模板（置顶，核心功能入口）──
    with st.expander("📋 风控分析模板（一键填充）", expanded=False):
        st.caption("点击模板自动填充专业分析Prompt")

        # 分组展示
        groups = {
            "📊 风险监控": ["📊 Vintage账龄分析", "📈 迁徙率滚动分析", "📋 通过率与拒绝率监控", "🚫 拒绝原因深度分析"],
            "🔬 模型变量": ["📐 WOE/IV分箱分析", "🔬 特征变量评估", "🎯 评分卡效果评估"],
            "🎯 策略决策": ["🔄 Swap置换分析", "🏆 冠军挑战者分析", "📋 三方数据源评估", "🔍 归因分析"],
            "💡 其他": ["💡 自由提问"],
        }

        for group_name, items in groups.items():
            st.caption(f"**{group_name}**")
            for label in items:
                content = TEMPLATES[label]
                display = label.replace("📊 ", "").replace("📈 ", "").replace("📋 ", "").replace("🚫 ", "").replace("📐 ", "").replace("🔬 ", "").replace("🎯 ", "").replace("🔄 ", "").replace("🏆 ", "").replace("🔍 ", "").replace("💡 ", "")
                if st.button(display, use_container_width=True, key=f"tpl_{label}"):
                    if label == "💡 自由提问":
                        st.session_state.pending_template = None
                    else:
                        st.session_state.pending_template = content

    st.divider()

    # ── 2. 数据上传 ──
    with st.expander("📎 上传数据文件", expanded=False):
        uploaded_file = st.file_uploader(
            "支持 CSV / Excel（.csv / .xlsx）",
            type=["csv", "xlsx"],
            key="file_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.uploaded_data = df
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.uploaded_columns = list(df.columns)

                st.success(f"✅ {uploaded_file.name}")
                st.caption(f"📊 {len(df):,} 行 × {len(df.columns)} 列")

                with st.expander("👁️ 数据预览", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"读取失败: {e}")

    # ── 3. 快速可视化 ──
    if st.session_state.uploaded_data is not None:
        with st.expander("📊 快速可视化", expanded=False):
            df = st.session_state.uploaded_data
            all_cols = list(df.columns)
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(exclude="number").columns.tolist()

            # 图表类型
            chart_type = st.selectbox(
                "图表类型",
                ["直方图", "箱线图", "折线图", "散点图", "柱状图", "饼图"],
                key="viz_chart_type",
            )

            if chart_type == "散点图":
                col_x = st.selectbox("X轴", all_cols, key="viz_x")
                col_y = st.selectbox("Y轴", all_cols, key="viz_y")
                color_col = st.selectbox("颜色分组（可选）", ["无"] + all_cols, key="viz_color")
                if st.button("生成散点图", use_container_width=True):
                    try:
                        fig = px.scatter(
                            df, x=col_x, y=col_y,
                            color=None if color_col == "无" else color_col,
                            title=f"{col_y} vs {col_x}",
                            opacity=0.6,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"生成失败: {e}")

            elif chart_type == "折线图":
                col_x = st.selectbox("X轴（时间/类别）", all_cols, key="viz_x")
                col_y = st.selectbox("Y轴（数值）", numeric_cols if numeric_cols else all_cols, key="viz_y")
                if st.button("生成折线图", use_container_width=True):
                    try:
                        fig = px.line(
                            df.sort_values(col_x) if col_x in df.columns else df,
                            x=col_x, y=col_y,
                            title=f"{col_y} 趋势图",
                            markers=True,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"生成失败: {e}")

            elif chart_type == "柱状图":
                col_x = st.selectbox("分类字段", cat_cols if cat_cols else all_cols, key="viz_x")
                top_n = st.slider("显示前N项", 5, 50, 15, key="viz_topn")
                if st.button("生成柱状图", use_container_width=True):
                    try:
                        counts = df[col_x].value_counts().nlargest(top_n).reset_index()
                        counts.columns = [col_x, "频次"]
                        fig = px.bar(counts, x=col_x, y="频次", title=f"{col_x} 分布 TOP{top_n}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"生成失败: {e}")

            elif chart_type == "饼图":
                col_x = st.selectbox("分类字段", cat_cols if cat_cols else all_cols, key="viz_x")
                top_n = st.slider("显示前N项", 3, 20, 8, key="viz_topn")
                if st.button("生成饼图", use_container_width=True):
                    try:
                        counts = df[col_x].value_counts().nlargest(top_n).reset_index()
                        counts.columns = [col_x, "频次"]
                        fig = px.pie(counts, names=col_x, values="频次", title=f"{col_x} 占比 TOP{top_n}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"生成失败: {e}")

            elif chart_type == "直方图":
                col_x = st.selectbox("数值列", numeric_cols if numeric_cols else all_cols, key="viz_x")
                nbins = st.slider("分箱数", 10, 100, 40, key="viz_bins")
                if st.button("生成直方图", use_container_width=True):
                    try:
                        fig = px.histogram(df, x=col_x, nbins=nbins, title=f"{col_x} 分布直方图", marginal="box")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"生成失败: {e}")

            elif chart_type == "箱线图":
                col_y = st.selectbox("数值列", numeric_cols if numeric_cols else all_cols, key="viz_y")
                col_x = st.selectbox("分组列（可选）", ["无"] + all_cols, key="viz_x")
                if st.button("生成箱线图", use_container_width=True):
                    try:
                        fig = px.box(
                            df, x=None if col_x == "无" else col_x, y=col_y,
                            title=f"{col_y} 箱线图",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"生成失败: {e}")

    # ── 4. 温度参数 ──
    with st.expander("🎚️ 温度参数（控制回答随机性）", expanded=False):
        st.caption("越低越严谨稳定（适合计算），越高越有创造性（适合策略建议）")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="0=极度严谨 | 0.3=平衡（推荐） | 1.0=创造性",
            label_visibility="collapsed",
        )
        if temperature <= 0.2:
            st.caption("🔒 当前: 极度严谨 — 适合数据计算与指标提取")
        elif temperature <= 0.5:
            st.caption("⚖️ 当前: 平衡 — 推荐用于风控分析")
        else:
            st.caption("🎨 当前: 创造性 — 适合策略思路发散")

    st.divider()

    # ── 5. 导出 & 清空 ──
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.session_state.messages:
            export_fmt = st.selectbox("导出格式", ["md", "xlsx", "csv"], label_visibility="collapsed", key="export_fmt")
            fmt_label = {"md": "📥 MD", "xlsx": "📥 Excel", "csv": "📥 CSV"}
            ext_map = {"md": "md", "xlsx": "xlsx", "csv": "csv"}
            mime_map = {"md": "text/markdown", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "csv": "text/csv"}
            data = export_conversation(st.session_state.messages, export_fmt)
            st.download_button(
                label=fmt_label[export_fmt],
                data=data,
                file_name=f"风控对话_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext_map[export_fmt]}",
                mime=mime_map[export_fmt],
                use_container_width=True,
            )
        else:
            st.button("📥 导出记录", disabled=True, use_container_width=True)

    st.divider()

    # ── 6. API 密钥配置（放在最底部，不显眼）──
    with st.expander("🔑 API 配置", expanded=False):
        selected_provider = st.selectbox(
            "AI 厂商",
            list(PROVIDERS.keys()),
        )

        env_key_name = PROVIDERS[selected_provider]["env_key"]
        env_val = os.getenv(env_key_name, "")
        api_key = st.text_input(
            f"API Key",
            type="password",
            value=env_val if env_val else "",
            placeholder=f"输入 Key 或设环境变量 {env_key_name}",
            key=f"api_key_{selected_provider}",
        )

        models_available = PROVIDERS[selected_provider]["models"]
        model = st.selectbox("模型", models_available, label_visibility="collapsed")

    st.caption("⚠️ AI分析仅供参考，不替代专业风控决策")

# ══════════════════════════════════════════════════════════════
# 主界面 - 聊天区
# ══════════════════════════════════════════════════════════════
st.title("🛡️ 金融风控智能分析助手")
st.caption(
    "上传数据 → 即刻分析 | Vintage & 迁徙率 & 策略监控 & 特征评估 & "
    "置换分析 & 冠军挑战者 | 支持 Claude / GPT / DeepSeek 多模型交叉验证"
)

# 数据已加载提示
if st.session_state.uploaded_data is not None:
    st.info(
        f"📎 已加载: **{st.session_state.uploaded_filename}** "
        f"({len(st.session_state.uploaded_data):,}行 × {len(st.session_state.uploaded_data.columns)}列) | "
        f"你的问题将基于此数据进行分析"
    )

# ── 渲染历史消息 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 用户输入 ──
input_value = st.session_state.pending_template
if input_value is not None:
    st.session_state.pending_template = None

prompt = st.chat_input("输入风控分析问题，或点击左侧模板一键填充…")

if not prompt and input_value:
    prompt = input_value

if prompt:
    if not api_key:
        st.error(f"❌ 请先在侧边栏底部「API 配置」中输入 {selected_provider} 的 API Key")
        st.stop()

    # 构建数据上下文
    data_context = ""
    if st.session_state.uploaded_data is not None:
        data_context = build_data_context(
            st.session_state.uploaded_data,
            st.session_state.uploaded_filename,
        )

    user_content = data_context + "\n---\n用户问题: " + prompt if data_context else prompt

    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_content})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 AI
    with st.chat_message("assistant"):
        with st.spinner(f"🧠 {selected_provider.split(' ')[0]} / {model} 分析中…"):
            try:
                reply = route_api_call(
                    provider=selected_provider,
                    model=model,
                    api_key=api_key,
                    messages=st.session_state.messages,
                    temperature=temperature,
                )
                st.markdown(reply)

                # 尝试识别 Plotly 代码块
                plotly_match = re.search(r"```python\s*(import plotly.*?fig\.show\(\))\s*```", reply, re.DOTALL)
                if plotly_match:
                    with st.expander("📊 查看图表代码"):
                        st.code(plotly_match.group(1), language="python")
                        st.caption("请复制上方代码在本地运行查看图表")

            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "invalid" in error_msg.lower() or "authentication" in error_msg.lower():
                    reply = f"❌ API Key 无效，请检查 {selected_provider} 的密钥。"
                elif "429" in error_msg:
                    reply = f"❌ 调用频率超限，请稍后重试。"
                elif "timeout" in error_msg.lower():
                    reply = f"❌ 请求超时，请重试。"
                else:
                    reply = f"❌ 调用失败: {error_msg}"
                st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
