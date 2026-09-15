# 这份稿子是怎么被检查的

三层，各抓不同的东西。前两层是机器，跑一次就够；第三层是人。

## 第一层：`verify_claims.py` —— 数字能不能重算

356 条。每条陈述一个论文里的数值，从 `results_*/` 重新算一遍，对不上就退出
非零。输入表缺失的记为 PENDING，同样算失败——所以一个数字不可能在还没法
重算的时候进入稿子。

```bash
python verify_claims.py            # 全部
python verify_claims.py --verbose  # 逐条打印
python verify_claims.py --todo     # 还在等表的检查
```

**它抓不到什么**：概括性断言。「All figures come from one code path」没有对应
的数字可以重算，所以 356 条一条都碰不到它——而那句话恰恰是错的（弃权分析和
评价单元重抽都不走那条路径）。

## 第二层：`scripts/analyse/cross_audit.py` —— 稿子内部自洽

```bash
python scripts/analyse/cross_audit.py --root . --manuscript manuscript_submission.md
```

五节：

1. **同名量取值是否一致。** 抓「N 个农场/session/病人/受试者/重抽次数/run」
   这类短语，把全部取值列出来。必然有误报（用第二台相机的三个农场 vs 八个
   农场），所以用基线机制：`--accept` 写入 `docs/cross_audit_baseline.json`，
   之后只报**新出现**的组合。
2. **交叉引用是否落空。** 正文引的 Supplementary Table/Note/Figure 是否存在、
   补充材料里有没有从未被引用的、Fig 有没有图注和图片文件、参考文献有没有
   列而未引或引而未列。
3. **三位小数能否在结果表里找到。** 粗筛错字和过期数字。
4. **投稿前必须清掉的东西。** TODO、格式备注、词数标注、占位符。
5. **篇幅。** 摘要、正文、display item 数对 NMI Analysis 的上限。

**这一层是被外部读者逼出来的。** 下面四处都是他们发现、而前两层漏掉的，
现在全部有机器检查盯着：

| 漏掉的 | 为什么漏 |
|---|---|
| Methods 说排除十二棵 Sabah 树，图注说显示十二棵 | 两处数字互相矛盾，各自都能从表里重算 |
| Methods 重抽 4,000 次，图注 2,000 次 | 同上 |
| Table 1 写 59 runs，正文却称 fully crossed | 数字与形容词矛盾 |
| 正文引用的 Supp Table 9–12 一度不存在 | 引用不是数字 |
| 十条参考文献列而未引 | 压缩正文时删掉了引用点 |

## 第三层：人 —— 概括是否跑在证据前面

机器查不了「这句话的证据覆盖了它声称的范围吗」。每改一版跑一次：

```bash
grep -nE "\b(all|every|none|never|only|identical|the same|cannot|no )" manuscript_submission.md
```

逐条问：这个样本量、这个设计，支持得起这个范围吗。

历史上被这一步抓到或应该被抓到的：

- 「selection-free estimate」—— 早停仍由留出单元决定，不是 selection-free
- 「the split rule is the only difference」—— durian 上训练集大小也不同
- 「weights that never saw it」—— 没进梯度更新，但进了模型选择
- 「mixing training sets can only add variation」—— 未经证明的数学命题
- 「nothing predicts which unit will fail」—— 八个农场的零结果，81 个病人推翻
- 「every unit-level figure is an upper bound」—— HAR 没有 checkpoint 选择

最后一条尤其值得记：**n=8 的零结果不是零结果，是没有功效。** 凡是「我们没
找到 X」的句子，先问这个样本量能不能找到 X。

## 每次改稿后的完整流程

```bash
python verify_claims.py
python scripts/analyse/cross_audit.py --root . --manuscript manuscript_submission.md
grep -nE "\b(all|every|none|never|only|identical|the same|cannot|no )" manuscript_submission.md
```

三条全绿再说下一步。新增一个量、一张表、一段结论时，问一句：
**这个量在两处出现了吗？如果是，有没有一条检查连着它们？**

## 第四层：文风 —— 有没有机器腔

词汇层面的 AI 标记（delve、underscore、pivotal、nuanced、testament、
In conclusion、Notably、It is worth noting）本来就没有。真正的机器腔在**结构
重复**：同一个句法动作反复做。

一次全文扫描的结果，改前对改后：

| 标记 | 改前 | 改后 |
|---|---|---|
| 冒号揭示（`陈述: 展开`） | 26 句 | 4 句 |
| `rather than` 对比 | 10 处 | 5 处 |
| 破折号插入语 | 5 处 | 4 处 |

26 句冒号意味着平均每五句就有一次同样的动作。单看每句都合理，连起来读者会
感到节奏机械。改法是拆成两句或改用 because/while，只在真正引出清单的地方
保留冒号。

也砍掉了几处格言式收尾。原稿几乎每段都想收一个警句（「confidence is most
useful where it is least needed」「the cross-region penalty is not there to
quantify」「evaluation units buy precision, not score」），单独看漂亮，连在
一起会让稿子显得比数据更确定。留最强的两三句，其余改朴素。

扫描命令：

```bash
python3 - <<'PY'
import re, statistics as st
t=open('manuscript_submission.md',encoding='utf-8').read()
b=t.split('## Main')[1].split('## Methods')[0]
s=re.split(r"(?<=[.!?])\s+", b.replace('\n',' '))
print('冒号揭示', sum(1 for x in s if re.search(r"[a-z]{4,}: [a-z]", x)))
print('rather than', b.count('rather than'))
print('破折号插入', len(re.findall(r"—[^—\n]{15,90}—", b)))
for p in (r"\b(?:delve|underscore[sd]?|pivotal|nuanced|multifaceted|testament)\b",
          r"\b(?:In conclusion|Overall|Furthermore|Moreover|Notably|Importantly)\b"):
    print(p[:30], len(re.findall(p, b)))
L=[len(re.findall(r"\S+",x)) for x in s if x.strip()]
print(f'句长 中位{st.median(L):.0f} 均值{st.mean(L):.1f} s.d.{st.pstdev(L):.1f}')
PY
```

句长中位 23、标准差 17 是健康的——长短句混着走。如果标准差掉到 8 以下，
说明每句都一样长，那也是机器腔的一种。
