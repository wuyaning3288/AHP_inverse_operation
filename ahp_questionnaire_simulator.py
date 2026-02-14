"""
AHP问卷模拟生成器 - Web交互版
基于你的新模拟.py改造
"""

import streamlit as st
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from io import BytesIO
import json

# Saaty RI table
RI_TABLE = {
    1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
    9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
}

def ahp_from_scores(scores, nos=None):
    """AHP计算：从scores生成判断矩阵并计算权重"""
    if nos is None:
        nos = list(range(len(scores)))
    items = list(zip(nos, scores))
    ranked = sorted(items, key=lambda x: (x[1], x[0]))
    n = len(ranked)

    A = np.ones((n, n), dtype=float)
    for i, (_, s_i) in enumerate(ranked):
        for j, (_, s_j) in enumerate(ranked):
            if i == j:
                A[i, j] = 1.0
            elif i > j:
                A[i, j] = float((s_i - s_j) + 1)
                A[j, i] = 1.0 / A[i, j]

    gm = np.prod(A, axis=1) ** (1 / n)
    w_ranked = gm / np.sum(gm)

    Aw = A.dot(w_ranked)
    lam = float(np.mean(Aw / w_ranked))
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.59)
    CR = (CI / RI) if RI != 0 else 0.0

    w_by_no = np.zeros(n)
    for i, (no, _) in enumerate(ranked):
        w_by_no[int(no)] = w_ranked[i]

    return w_by_no, lam, CI, CR

def compositions(n, k=5):
    """把n分成k份的所有非负整数组合"""
    if k == 1:
        yield (n,)
        return
    for x in range(n + 1):
        for rest in compositions(n - x, k - 1):
            yield (x,) + rest

def gen_candidates(target_weights, top_keep=120):
    """枚举所有分档形态，生成候选问卷"""
    t = np.array(target_weights, dtype=float)
    if t.sum() <= 0:
        raise ValueError("target_weights sum must be > 0")
    t = t / t.sum()
    n = len(t)

    order = np.argsort(t)
    cands = []
    for cnt in compositions(n, 5):
        levels = []
        for score, ct in enumerate(cnt, start=1):
            levels += [score] * ct
        if len(levels) != n:
            continue

        scores = [None] * n
        for idx, ind_idx in enumerate(order):
            scores[ind_idx] = levels[idx]

        w, lam, CI, CR = ahp_from_scores(scores, nos=list(range(n)))
        dist = float(np.linalg.norm(w - t))
        cands.append((dist, scores, w, CR, lam))

    cands.sort(key=lambda x: x[0])
    return cands[:top_keep], t

def consistency_label(cr, thr=0.1):
    return "通过" if cr < thr else "不通过"

def best_k_mean(target_weights, k=3, top_keep=80, cr_threshold=0.1, beam_width=200, allow_replacement=False):
    """用beam search选k份问卷，使平均权重最接近target"""
    pool, t = gen_candidates(target_weights, top_keep=top_keep)
    pool = [p for p in pool if p[3] <= cr_threshold]
    if not pool:
        raise ValueError("没有候选问卷满足CR阈值要求，请增加top_keep或放宽cr_threshold")

    W = np.stack([p[2] for p in pool], axis=0)
    target_sum = k * t

    beam = [(float(np.linalg.norm(target_sum)), np.zeros_like(t), tuple())]
    for _ in range(k):
        new_beam = []
        for _, svec, idxs in beam:
            if allow_replacement:
                candidates = range(len(pool))
            else:
                used = set(idxs)
                candidates = [i for i in range(len(pool)) if i not in used]
                if not candidates:
                    continue
            for i in candidates:
                ns = svec + W[i]
                nidxs = idxs + (i,)
                dist = float(np.linalg.norm(ns - target_sum))
                new_beam.append((dist, ns, nidxs))

        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        if not beam:
            raise ValueError("Beam search失败（k可能太大）")

    best_dist, best_sum, best_idxs = beam[0]
    chosen = [pool[i] for i in best_idxs]
    mean_w = best_sum / k
    return best_dist, chosen, mean_w, t

def build_module_df_k(target_weights, k=3, top_keep=80, cr_threshold=0.1, beam_width=200, allow_replacement=False):
    """构建单个模块的DataFrame结果"""
    dist, chosen, mean_w, t = best_k_mean(
        target_weights, k=k, top_keep=top_keep, cr_threshold=cr_threshold,
        beam_width=beam_width, allow_replacement=allow_replacement
    )
    n = len(t)
    df = pd.DataFrame({"No": list(range(1, n + 1)), "Target weight": t})

    crs = [c[3] for c in chosen]
    overall = "通过" if all(cr <= cr_threshold for cr in crs) else "不通过"

    for qi, cand in enumerate(chosen, start=1):
        df[f"Q{qi} Score"] = cand[1]
        df[f"Q{qi} Weight"] = cand[2]
        df[f"Q{qi} CR"] = [cand[3]] * n

    df["一致性检验是否通过(CR<0.1)"] = [overall] * n
    df["Mean weight"] = mean_w
    df["Abs err"] = np.abs(mean_w - t)

    summary = {
        "k": k,
        "best_mean_L2": float(dist),
        "mean_abs_error": float(df["Abs err"].mean()),
        "max_abs_error": float(df["Abs err"].max()),
        "Overall": overall,
    }
    for qi, cr in enumerate(crs, start=1):
        summary[f"Q{qi}_CR"] = float(cr)
        summary[f"Q{qi}_pass"] = consistency_label(cr, cr_threshold)
    return df, summary

def write_df_to_sheet(ws, df, summary, start_row=5):
    """写入Excel sheet"""
    # summary
    ws["A1"] = "k"; ws["B1"] = summary["k"]
    ws["D1"] = "best_mean_L2"; ws["E1"] = summary["best_mean_L2"]
    ws["G1"] = "mean_abs_error"; ws["H1"] = summary["mean_abs_error"]
    ws["J1"] = "max_abs_error"; ws["K1"] = summary["max_abs_error"]
    ws["M1"] = "Overall"; ws["N1"] = summary["Overall"]

    col = 1
    for qi in range(1, summary["k"] + 1):
        ws.cell(2, col).value = f"Q{qi} CR"
        ws.cell(2, col+1).value = summary[f"Q{qi}_CR"]
        ws.cell(2, col+2).value = summary[f"Q{qi}_pass"]
        col += 4

    # header
    for j, colname in enumerate(df.columns, start=1):
        ws.cell(start_row, j, colname)

    # data
    for i, rowvals in enumerate(df.itertuples(index=False), start=start_row + 1):
        for j, val in enumerate(rowvals, start=1):
            ws.cell(i, j, val)

    # styling
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    bold = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin = Side(style="thin", color="808080")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cell in ws[start_row]:
        cell.font = bold
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border

    for r in range(start_row + 1, start_row + 1 + len(df)):
        for c in range(1, len(df.columns) + 1):
            cell = ws.cell(r, c)
            cell.alignment = center
            cell.border = border

    # formats
    percent_cols = []
    for j, name in enumerate(df.columns, start=1):
        if name in ("Target weight", "Mean weight", "Abs err") or name.endswith(" Weight"):
            percent_cols.append(j)
    for r in range(start_row + 1, start_row + 1 + len(df)):
        for j in percent_cols:
            ws.cell(r, j).number_format = "0.00%"

    for j, name in enumerate(df.columns, start=1):
        if name.endswith(" CR"):
            for r in range(start_row + 1, start_row + 1 + len(df)):
                ws.cell(r, j).number_format = "0.000"

    for addr in ["E1","H1","K1"]:
        ws[addr].number_format = "0.000000"

def generate_excel_bytes(modules: dict, k=3, top_keep=80, cr_threshold=0.1, beam_width=200, allow_replacement=False):
    """生成Excel并返回bytes"""
    wb = Workbook()
    wb.remove(wb.active)
    for sheet_name, target_weights in modules.items():
        df, summary = build_module_df_k(
            target_weights, k=k, top_keep=top_keep,
            cr_threshold=cr_threshold, beam_width=beam_width,
            allow_replacement=allow_replacement
        )
        ws = wb.create_sheet(title=str(sheet_name)[:31])
        write_df_to_sheet(ws, df, summary)
    
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    return output

# ==================== Streamlit UI ====================

st.set_page_config(page_title="AHP问卷模拟生成器", page_icon="🎯", layout="wide")

st.title("🎯 AHP问卷模拟生成器")
st.markdown("自动生成k份问卷，使平均权重最接近目标权重")
st.markdown("---")

# 侧边栏：全局参数
st.sidebar.header("⚙️ 全局参数")
k_value = st.sidebar.slider("问卷份数 (k)", min_value=2, max_value=10, value=4, 
                             help="生成多少份问卷")
top_keep = st.sidebar.number_input("候选池大小", min_value=50, max_value=300, value=80,
                                    help="保留多少个候选问卷")
cr_threshold = st.sidebar.slider("CR阈值", min_value=0.05, max_value=0.15, value=0.1, step=0.01,
                                  help="一致性检验阈值")
beam_width = st.sidebar.number_input("Beam宽度", min_value=100, max_value=500, value=250,
                                      help="Beam search的搜索宽度")
allow_replacement = st.sidebar.checkbox("允许重复选择同一问卷", value=False)

# 主界面：模块输入
st.header("📝 模块配置")

# 初始化session state
if 'modules' not in st.session_state:
    st.session_state.modules = {
        "模块1_一级指标": [0.1411, 0.4550, 0.2627, 0.1411],
        "模块2_示例": [0.08, 0.06, 0.12, 0.15, 0.06, 0.08, 0.10, 0.05, 0.05, 0.05]
    }

# 选项卡：输入方式
tab1, tab2 = st.tabs(["📋 表格输入", "💻 JSON输入"])

with tab1:
    st.markdown("### 当前模块")
    
    # 显示和编辑现有模块
    modules_to_delete = []
    for module_name in list(st.session_state.modules.keys()):
        with st.expander(f"🗂️ {module_name}", expanded=False):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                weights = st.session_state.modules[module_name]
                
                # 显示权重
                st.write(f"**指标数量：** {len(weights)}")
                
                # 编辑权重
                weight_cols = st.columns(min(5, len(weights)))
                new_weights = []
                for i, w in enumerate(weights):
                    with weight_cols[i % 5]:
                        new_w = st.number_input(
                            f"指标{i+1}", 
                            value=float(w), 
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            format="%.4f",
                            key=f"weight_{module_name}_{i}"
                        )
                        new_weights.append(new_w)
                
                # 更新权重
                st.session_state.modules[module_name] = new_weights
                
                # 显示总和
                total = sum(new_weights)
                if abs(total - 1.0) > 0.01:
                    st.warning(f"⚠️ 权重总和：{total:.4f} (建议为1.0)")
                else:
                    st.success(f"✅ 权重总和：{total:.4f}")
            
            with col2:
                if st.button("🗑️ 删除", key=f"del_{module_name}"):
                    modules_to_delete.append(module_name)
    
    # 执行删除
    for module_name in modules_to_delete:
        del st.session_state.modules[module_name]
        st.rerun()
    
    st.markdown("---")
    
    # 添加新模块
    st.markdown("### 添加新模块")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        new_module_name = st.text_input("模块名称", value="新模块", key="new_module_name")
    
    with col2:
        num_indicators = st.number_input("指标数量", min_value=2, max_value=20, value=4, key="num_indicators")
    
    with col3:
        st.write("")  # spacing
        st.write("")  # spacing
        if st.button("➕ 添加模块", type="primary", use_container_width=True):
            if new_module_name and new_module_name not in st.session_state.modules:
                # 创建均匀权重
                uniform_weight = 1.0 / num_indicators
                st.session_state.modules[new_module_name] = [uniform_weight] * num_indicators
                st.success(f"✅ 已添加模块：{new_module_name}")
                st.rerun()
            else:
                st.error("❌ 模块名称重复或为空")

with tab2:
    st.markdown("### JSON格式输入")
    st.markdown("可以直接粘贴JSON格式的模块配置")
    
    # 显示当前JSON
    current_json = json.dumps(st.session_state.modules, ensure_ascii=False, indent=2)
    
    json_input = st.text_area(
        "模块配置 (JSON格式)",
        value=current_json,
        height=300,
        help="格式: {\"模块名\": [权重1, 权重2, ...]}"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("📥 加载JSON", type="primary"):
            try:
                new_modules = json.loads(json_input)
                # 验证格式
                for name, weights in new_modules.items():
                    if not isinstance(weights, list):
                        raise ValueError(f"模块 {name} 的权重必须是列表")
                    if not all(isinstance(w, (int, float)) for w in weights):
                        raise ValueError(f"模块 {name} 的权重必须是数字")
                
                st.session_state.modules = new_modules
                st.success("✅ JSON加载成功！")
                st.rerun()
            except Exception as e:
                st.error(f"❌ JSON格式错误: {str(e)}")

# 生成按钮和结果
st.markdown("---")
st.header("🚀 生成问卷")

col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    st.metric("模块数量", len(st.session_state.modules))

with col2:
    total_indicators = sum(len(w) for w in st.session_state.modules.values())
    st.metric("总指标数", total_indicators)

with col3:
    if st.button("🎲 生成问卷", type="primary", use_container_width=True):
        if not st.session_state.modules:
            st.error("❌ 请至少添加一个模块")
        else:
            with st.spinner("正在生成问卷..."):
                try:
                    # 生成Excel
                    excel_bytes = generate_excel_bytes(
                        st.session_state.modules,
                        k=k_value,
                        top_keep=top_keep,
                        cr_threshold=cr_threshold,
                        beam_width=beam_width,
                        allow_replacement=allow_replacement
                    )
                    
                    st.success("✅ 问卷生成成功！")
                    
                    # 下载按钮
                    st.download_button(
                        label="📥 下载Excel文件",
                        data=excel_bytes,
                        file_name=f"AHP问卷模拟_{k_value}份.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    # 显示预览
                    st.markdown("---")
                    st.subheader("📊 生成结果预览")
                    
                    for module_name, target_weights in st.session_state.modules.items():
                        with st.expander(f"🗂️ {module_name}", expanded=True):
                            try:
                                df, summary = build_module_df_k(
                                    target_weights,
                                    k=k_value,
                                    top_keep=top_keep,
                                    cr_threshold=cr_threshold,
                                    beam_width=beam_width,
                                    allow_replacement=allow_replacement
                                )
                                
                                # 显示汇总指标
                                metric_cols = st.columns(4)
                                metric_cols[0].metric("平均绝对误差", f"{summary['mean_abs_error']:.6f}")
                                metric_cols[1].metric("最大绝对误差", f"{summary['max_abs_error']:.6f}")
                                metric_cols[2].metric("L2距离", f"{summary['best_mean_L2']:.6f}")
                                metric_cols[3].metric("一致性检验", summary['Overall'])
                                
                                # 显示表格
                                st.dataframe(df, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"❌ 模块 {module_name} 生成失败: {str(e)}")
                    
                except Exception as e:
                    st.error(f"❌ 生成失败: {str(e)}")

# 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 如何使用？
    
    **1. 配置模块**
    - 在"模块配置"区域添加你的模块
    - 每个模块包含：模块名称 + 目标权重列表
    - 目标权重建议总和为1.0
    
    **2. 设置参数**
    - 问卷份数(k)：生成多少份问卷
    - CR阈值：一致性检验标准（默认0.1）
    - 其他参数保持默认即可
    
    **3. 生成问卷**
    - 点击"生成问卷"按钮
    - 系统会自动找到k份问卷，使其平均权重最接近目标
    - 下载Excel文件查看完整结果
    
    ### 算法原理
    
    系统会：
    1. 枚举所有可能的1-5分评分组合
    2. 计算每个组合对应的AHP权重
    3. 用beam search找到最优的k份组合
    4. 确保所有问卷的一致性检验都通过（CR<0.1）
    
    ### 示例
    
    **目标权重**: [0.1411, 0.4550, 0.2627, 0.1411]  
    **k=4**: 生成4份问卷  
    
    系统会输出：
    - 每份问卷的1-5分评分
    - 每份问卷计算出的权重
    - 4份问卷的平均权重
    - 平均权重与目标权重的误差
    """)

st.markdown("---")
st.caption("💡 AHP问卷模拟生成器 | 基于Beam Search优化")
