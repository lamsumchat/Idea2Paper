# Agentic Search 设计方案

> 日期：2026-02-17
> 分支：`feature/agentic-search`
> 状态：v1 实现完成

## 概述

在现有三路静态召回之后，根据召回质量自适应触发联网搜索，搜索 CS/AI 顶会论文并深度注入 Pattern 体系。

## 阶段 1：静态召回

现有三路 Recall 不变，输出 Top-10 Pattern（基于 ICLR 2025 KG）。

## 阶段 2：自适应决策

计算综合质量分数决定搜索力度：

```
avg_score = (score₁×0.5 + score₂×0.3 + score₃×0.2) × 0.7
          + (Σ score₁₋₁₀ / 10) × 0.3
```

| avg_score | 搜索篇数 | 年份阈值 |
|-----------|---------|---------|
| > 0.7 | 5 篇 | ≥ 2025 |
| 0.4 ~ 0.7 | 10 篇 | ≥ 2024 |
| < 0.4 | 20 篇 | ≥ 2023 |

分层搜索配额（按年份分配）：

- 2025 年：5 篇
- 2024 年：3 篇
- 2023 年：2 篇

（按年份阈值截断，如 `≥ 2025` 则只搜 2025 年的 5 篇配额）

## 阶段 3：Agentic Search

### 数据源

| 优先级 | MCP | 作用 |
|--------|-----|------|
| 主要 | mcp-dblp | 按会议+年份精确搜索 CS/AI 顶会论文 |
| 辅助 | Semantic Scholar MCP | 补充 abstract、引用链分析 |

配合逻辑：DBLP 搜到论文 title/venue/year → Semantic Scholar 获取 abstract。

### Agent 推理循环（max 2 轮）

```
Round 1:
  LLM 从用户 idea 分解搜索意图 → 生成多组 query
  → mcp-dblp 搜索（venue_filter 限定顶会）
  → Semantic Scholar 补充 abstract
  → 基础过滤（去除无 abstract / 非英文 / 明显无关）
  → 快速排序：embedding(user_idea) vs embedding(title+abstract)
  → 截取 Top-N

质量检查：
  Top-N 平均相似度 > 阈值 → 满足，结束
  Top-N 平均相似度 < 阈值 → Round 2

Round 2:
  LLM 分析 Round 1 失败原因（太宽泛 / 太窄 / 会议覆盖不够）
  → 生成改进后的 query
  → 搜索 → 排序 → 合并 Round 1 + Round 2 去重结果
```

## 阶段 4：三层去重

1. **Paper ID 去重**：DBLP key / DOI 精确匹配，去除与 KG 已有论文的重复
2. **语义去重**：embedding 相似度 > 0.95 的论文对只保留更新的那篇
3. **年份过滤**：严格执行阶段 2 确定的年份阈值

## 阶段 5：智能 Pattern 提取

### 前置步骤：LLM 结构化提取

每篇论文一次 LLM 调用，提取以下字段（与现有 KG 节点格式对齐）：

- `idea_summary`
- `problem_definition`
- `solution_pattern`
- `domain` / `sub_domains`

### 快速筛选

abstract embedding vs 全部 124 个 Pattern 的 embedding，取 Top-20 候选 Pattern。

### 精确计算

对 Top-20 候选做多特征加权相似度：

```
similarity =
  abstract_emb  vs Pattern summary         × 0.4
  problem_def   vs Pattern base_problem    × 0.25
  solution      vs Pattern solution_pattern × 0.25
  domain/sub_domain 文本匹配               × 0.1
```

### 分流判断

| 相似度 | 处理方式 |
|--------|---------|
| ≥ 0.85 | 归入现有 Pattern，提升其得分 |
| 0.75 ~ 0.85 | 灰色区间，跳过 |
| < 0.75 | 标记为"待聚类" |

## 阶段 6：批量聚类

对"待聚类"论文：

- **≤ 2 篇**：直接 LLM 提取，每篇生成 1 个动态 Pattern
- **≥ 3 篇**：LLM 直接分组 + 每组提取 1 个 Pattern（fallback: DBSCAN 聚类）
- **质量过滤**：质量 < 0.5 的动态 Pattern 剔除

## 阶段 7：合并结果

静态 Pattern 得分 + 动态 Pattern 得分，应用防护机制：

- 动态 Pattern 最多 **3 个**
- 动态 Pattern 权重 **cap at 0.6**（防止动态结果压过静态知识）
- 排序输出 Top-10

## 阶段 8：后续流程不变

Top-10 Pattern → Pattern Selection → Story Generation → Refinement → ...

## 配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `AGENTIC_SEARCH_ENABLE` | `false` | 总开关 |
| `AGENTIC_SEARCH_SOURCES` | `["dblp", "semantic_scholar"]` | 启用数据源 |
| `AGENTIC_SEARCH_MAX_ROUNDS` | `2` | 最大迭代轮数 |
| `AGENTIC_SEARCH_RESULTS_PER_ROUND` | `10` | 每轮每源返回数 |
| `AGENTIC_SEARCH_FINAL_TOP_K` | `5` | 最终保留论文数 |
| `AGENTIC_SEARCH_RELEVANCE_THRESHOLD` | `0.6` | 相关性阈值 |
| `AGENTIC_SEARCH_RECALL_WEIGHT` | `0.15` | 动态 Pattern 融合权重 |

## 技术选型

| 组件 | 方案 |
|------|------|
| 主数据源 | mcp-dblp（DBLP，CS 顶会精确搜索） |
| 辅数据源 | Semantic Scholar MCP（补充 abstract + 引用链） |
| 集成方式 | MCP 协议调用 |
| Agent 推理 | 多轮迭代（max 2 轮），含质量检查 + 针对性重试 |
| 搜索排序 | embedding 快速排序（每轮内） |
| 结构化提取 | LLM 从 abstract 提取 problem/solution/domain |
| 精确匹配 | 多特征加权相似度（4 维） |
| 聚类 | LLM 直接分组（fallback: DBSCAN） |
| 注入方式 | 深度注入 Pattern 体系 |
| 作用范围 | CS/AI 顶会论文，邻近领域补充 |

## 设计边界

- 不做完全跨域（文学、历史等），KG 扩充交给团队长期规划
- 不替代现有三路 Recall，作为后置补充
- 如果 `AGENTIC_SEARCH_ENABLE=false`，系统行为和现在完全一致（三路权重回归原始 0.4/0.2/0.4）
