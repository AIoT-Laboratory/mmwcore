# GTrack3D 能力与 TI 源码版本审计

审计日期：2026-09-05。范围是本机 Radar Toolbox 4.00.00.05 中 **6843 工程实际使用的 custom SDK3 trackerproc_overhead GTRACK**，不是任意同名 GTRACK 库的能力并集。对照的 mmwcore HEAD 为 `e799600a0813ae0d8ea26a30a71d5e84c5828b56`，审阅时其 GTrack3D 为 Rust 6D 实现。本次只读源代码与既有构建材料，没有重跑旧研究实验、采集硬件或验证新实现。

## 结论

当前 mmwcore 已有逐点预测关联、扩展目标 dispersion、球面 EKF 和生命周期，但不等价于这一版 stock GTRACK。缺口涉及状态模型、关联资格、centroid 选点、速度展开、静态状态处理和事件计数，不能靠扩大 gate 或延长 timeout 补齐。

最可靠的完整基线策略是：**固定本机官方源码及哈希，提供一个明确标识 TI 版本的完整 C 后端，通过 Rust 所有权与参数检查边界接入 Python**；原先 6D Rust 基线保留独立标识。若必须取得完全独立的 Rust 算法实现，则应把“逐项差分达到版本行为一致”作为另一项移植工作，不能仅凭功能名称相似宣布 stock parity。此建议是工程判断，不是实测性能结论。

附件中两项会改变研究创新点表述的概括需要修正：本版本已经区分“关联点”和“可更新状态的可靠点”；本版本生命周期也不只是“有任意关联点即 HIT”。后续 RT 研究应对照这些已有机制，而不是对照一个全部点等权更新的概念简化版。

## 1. 锁定实际参考版本

- 6843 MSS 工程定义 `GTRACK_3D`，从 `MODIFIED_SDK3_DATAPATH/dpu/trackerproc_overhead/packages/ti/alg/gtrack/lib` 链接 `libgtrack3D.aer4f`；其 tracker_utils 把 `stateVectorType` 设为 `GTRACK_STATE_VECTORS_3DA`。[MSS 工程:52][project]、[tracker_utils:274][utils]
- 本地来源根目录：`C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack`。
- 既有 allocation 实验的 [build.json][build] 记录了同一来源目录、`-DGTRACK_3D`、关闭 fast-math 和浮点收缩的 host 构建，以及 23 个 TI 源/头文件 SHA-256。本次逐一重算 23 项，全部与记录相符。
- [build_reference.py][builder] 将完整 TI 源码编入库，但它的研究 wrapper 仅调用 allocation；不能把“曾成功编译完整源码”写成“已有完整 gtrack_step 行为验证”。[host_reference.c:77][host]
- TI 官方 [3D People Tracking User Guide][guide] 指向 6843、SDK3.5 和本例程；[TIDUE71][design] 是参考设计背景。附件引用的 [Jacinto PTK API][ptk] 可解释概念，但不是本机 custom SDK3 的逐行行为依据。

## 2. 实际 stock 能力与当前 mmwcore 差距

| 能力 | 本版本实际行为 | 当前 Rust GTrack3D 差距 |
| --- | --- | --- |
| 帧阶段顺序 | setup → Predict → Associate → Allocate → Update → Presence → Report；新分配 unit 在同帧进入 Update。[step:198][step] | 预测→关联→更新→miss/delete→allocation，出生帧计数及何时可重用点不同。[measurement3d.rs:284][current-step] |
| 运动状态 | 3DA 是 9 维位置、速度、加速度；静态目标预测时保持状态和协方差。侧装、顶装有不同策略。[unit_create:133][create]、[unit_predict:83][predict] | 6D constant velocity；acceleration 参数仅驱动过程噪声，没有加速度状态，也没有 stock static 状态冻结。[measurement3d.rs:107][current-filter] |
| 边界/静态点资格 | 世界坐标边界筛点；有 static box 时，零 Doppler 点还须进入 static box。顶装有额外 boresight 筛选分支。[step:131][step] | 只有普通 scenery 边界关联筛选；static box 主要决定 coast 时限。[measurement3d.rs:307][current-assoc]、[measurement3d.rs:684][current-life] |
| 关联 gate/score | 先按测量维度的物理 limits；partial Mahalanobis gate 忽略 Doppler，动态点/静态 unit 使用不同 gate；score 再把 Doppler residual 乘 3 后做 full Mahalanobis。[unit_score:129][score] | Cartesian 总距离 gate + 径向速度差 gate + full Mahalanobis gate；并非 stock 的 gate/score 拆分。[measurement3d.rs:307][current-assoc] |
| 点资格 | bestIndex、bestScore、unique bitmap 和 static bitmap；竞争点的 unique 清除有动态对静态等例外。[unit_score:293][score] | 输出唯一获胜 track id，但无竞争唯一性位和 stock qualification。[measurement3d.rs:350][current-assoc] |
| 状态更新点 | 仅 dynamic AND unique 点形成 reliable/good 集合；侧装 SNR 加权、顶装算术均值。静态点和非 unique 动态点仍计入存在/状态逻辑。[unit_update:155][update] | 所有关联点一起形成四维算术均值，没有上述点资格拆分。[measurement3d.rs:755][current-summary] |
| extended-target 统计 | 限幅、非对称平滑的 spread；估计目标点数；默认测量方差来自 spread；以 good 点更新 dispersion；Rc 采用显式点数缺失系数和 dispersion 对角项。[unit_update:213][update]、[unit_update:400][update] | 固定 point noise、不同 expected_points 平滑、不同 missing² 系数及 full dispersion；不能通过复用参数名认为等价。[measurement3d.rs:419][current-update]、[measurement3d.rs:820][current-summary] |
| 速度展开 | INIT → RATE_FILTER → TRACKING → LOCKED；根据从 allocation 至今的 range rate 与预测 Doppler 逐阶段展开。[unit_update:660][update] | 每点/均值就近展开到当前预测，缺少 range-rate 初始化和状态机。[measurement3d.rs:884][current-summary] |
| static / dynamic | 无点或仅静态点时根据世界 XY 速度冻结或减速；可靠动态点和 Doppler 条件控制恢复；confidence 随证据变化。[unit_update:284][update] | 没有该内部状态及 confidence；coasting + 低组均值 Doppler 的特殊规则只更新 EKF、拒绝生命周期 hit。[measurement3d.rs:407][current-update] |
| allocation | 按输入点顺序尝试种子，每加入一点更新 centroid；位置距离与速度差为两个独立门限；仅选最大点数集合，最多一条新轨迹；range/obscured/static-zone 条件改变 SNR 阈值。[module:156][module] | 按 SNR 排序并以固定 lead 分组，XY 与速度在同一距离中混合；candidate 再按点数/SNR 排序；没有 stock 的 range/obscured SNR 规则。[measurement3d.rs:487][current-alloc]、[measurement3d.rs:701][current-life] |
| lifecycle | DETECTION/ACTIVE 的可靠点数、动态点数、静态状态、区域、confidence、sleep history 有各自逻辑，阈值比较也不同。[unit_event:72][event] | Tentative/Confirmed/Coasting + min_update_points + miss limit 是另一种策略。[measurement3d.rs:369][current-update]、[measurement3d.rs:658][current-life] |
| 输出 | unit uid、递增 target tid、9D S、EC、G、dim、uCenter、confidence；逐点索引及 unique bitmap；presence 输出。[unit_report:72][report]、[step:225][step] | 位置/速度及位置和 extent 协方差、status/age/miss、point ids、累计诊断；缺少大部分 stock 输出。[measurement3d.rs:622][current-life] |

## 3. 不能混用的版本细节

### dynamicSNRcentroid 与 membership / update 的区分

在本机整个 `custom_sdk_files` 的 C/H 搜索中未找到 `dynamicSNRcentroid` 标识。本版本开关是内部 `isSnrWeighting`，不是名为 dynamicSNRcentroid 的公开 CLI 或 API 参数。

3DA 侧装打开 SNR weighting 与 estimated point count；仰俯角距离 90° 小于 20.5° 的顶装分支关闭二者，并开启 association height-ignore。两种分支都只用 dynamic AND unique 点更新 centroid。[unit_create:133][create]、[unit_update:166][update]

“dynamic”在这些函数中基本指绝对 Doppler 大于浮点 epsilon，**不是人体整体速度大于 0.15 m/s**。因此这套 qualification 也不能证明已识别出躯干/手臂的功能贡献，只能说 stock 已有一层 association 与 state-update 资格分工。

### Ghost 支持存在，但 6843 3DA 默认不启用

unitScore 包含 ghost-behind 和 likely-multipath 分支；不过 unitCreate 在 2DA 开启、3DA 关闭 `isAssociationGhostMarking`。因此“库源码包含 ghost 标记”与“本例程的 stock 3D 在运行该机制”必须区分。[unit_create:119][create]、[unit_create:140][create]、[unit_score:201][score]

即使 ghost marking 关闭，unique/static 竞争资格机制仍在，不能把三者一起当作不存在。

### HIT/MISS 与 > / >= 是实际行为的一部分

DETECTION 下：

- `numReliable > 3` 才增加确认计数；确认条件是 count **大于** det2actThre。
- `numReliable == 0` 才增加 detect2freeCount，并将确认计数减一（非清零）。
- 1–3 reliable 点会清除检测失败计数，但不增加确认计数。

ACTIVE 下，static target 任意关联点可以 HIT；dynamic target 需要至少动态点。sleep2free 还有静态点历史累计带来的清零，以及 confidence/区域决定的更短阈值。普通 active/static/exit 删除以 miss count **大于**对应阈值触发；边界外删除另用 `>=`。[unit_event:100][event]

附件中的“至少一个 associated point 是统一 HIT”和“连续 HIT/MISS 的普通状态图”只能算概念层说明，不足以写兼容测试。

### EC 名称不能直接解释为状态协方差

头文件将 EC 描述为 group covariance，但本版 unitScore 结尾把 `gC_inv` 复制到内部 `ec`，unitReport 再复制到 EC。因此该版本实际报告的是对应时刻的 **4×4 group inverse covariance**，不是 9×9 状态 P，也不是 Cartesian position covariance。[unit_score:351][score]、[unit_report:80][report]

moduleReport 遍历 activeList，**没有筛除 DETECTION**。activeList 是分配后仍存活的 unit 集合，不等同于 TrackState ACTIVE。若 OpenMMW 仍只显示 confirmed/coasting，需要由适配器明确保留内部状态，再做展示筛选。[module:564][module]

uid 是可复用 unit 槽；tid 是目标身份计数；逐点 mIndex 使用 uid。接入研究输出时必须明确转换，不能把 uid 复用计成同一人持续 ID，也不能误报切换数量。[gtrack.h:687][api]、[module:380][module]

## 4. 建议的完整实现和验收边界

这是实施建议，未在本审计中执行：

1. **明确版本后端。** 为当前 TI-device 数据提供来源固定的 stock backend；从配置和输出 metadata 一直保留版本、源哈希、编译选项、mount 模式和全部实际参数。不要偷偷将旧 Rust 的数值参数套入同名 stock 字段。
2. **固定完整 gtrack_step。** 编译外部官方源文件，通过窄 C ABI 接入 Rust；Rust 管理实例所有权、生命周期、输入长度/有限值/单位/参数检查和清晰错误。现有 allocation observer 只作为来源与编译线索，不能充当完整正式 backend。
3. **暴露完整输入/输出。** 支持 spherical point + 线性 SNR（头文件明确为 linear）、可选 measurement variance、mount/scenery/gating/allocation/state/presence 参数；输出 S9、原始 EC4×4 与其明确语义、G/dim/uCenter/confidence、uid/tid、association/unique、内部状态与必要诊断。[gtrack.h:668][api]
4. **保持研究接口区分。** 旧 6D 算法与新 TI stock 算法使用不同 model id；RT evidence hooks 此阶段不进入 stock 数值路径。OpenMMW 展示投影可兼容，但原始 stock 输出应可读取。
5. **验收完整行为，不只检查可运行。** 对选定官方原始 gtrack_step 建立确定性 CPU 序列，覆盖出生帧阶段顺序、输入排列/最大集合 tie、1/3/4 reliable 点、unique 竞争、侧装与顶装、static→dynamic、无点与仅静态点、sleep/exit、速度折叠、uid 复用、presence。逐帧比较 S/P（若暴露内部只读快照）/association/unique/计数，明确 float32 容差及边界条件。
6. **双重限制分别说明。** “对官方实现数值一致”证明实现保真；不证明当前雷达 RPC 前端与 TI Capon 前端相同，也不证明本项目人体跟踪精度提高。Doppler 正负及坐标/SNR 契约需要独立一致性验证，不能用引入 stock 后的改观倒推旧符号必然错误。

若选择原生 Rust 迁移，至少以上能力表每行都需有可追溯的实现/差分证据，并单独处理官方 float32 分支的边界行为。一次大幅重写同时修复旧算法与引入 RT，会使结果归因失去清晰性。

## 5. TI 源码许可与分发边界

以下是所安装源码许可文本的直接事实与工程含义；它不替代针对具体发布方式的法律判断：

- 这些文件采用 TI Limited License，要求保留版权与许可；源码、修改及衍生作品，以及编译后的对象代码，其 use / redistribution 均限制为与 TI Devices 配合使用。[gtrack.h:1][license]
- 许可原文的关键短语是 “for use only with TI Devices”。把实现转写为 Rust，不会仅因语言变化就自动获得 Apache-2.0 的无限制再许可。
- mmwcore 工作区当前声明 `Apache-2.0`。因此把 TI 源码或其派生移植直接混入包、却让整个分发看起来只有 Apache 许可，会掩盖实际限制。[Cargo.toml:10][cargo]
- **最保守的工程选择**是默认包不捆绑 TI 源码/DLL，用户指定自己已安装的官方 source root，编译/加载显式受限后端；构建 provenance 保留许可文本和源哈希，公开元数据声明 TI-device 限制。外部依赖方式不抹除许可限制，只使来源与分发边界更清楚。
- 若选择随包分发源或二进制，需保留 TI 原始许可并明确包内第三方文件、受限适用范围，不能将该后端宣传为适用于任意雷达的通用 Apache GTRACK。

## 6. 完整桥接实现的独立 source / ABI 复核

此节是在上述审计后，对新增 `tools/ti_gtrack/bridge.c/.h/build.py` 与 `crates/mmwcore-ti-gtrack/src/lib.rs` 所做的第二轮有界审阅。只在 gitignored 的 `build/ti-gtrack-review` 写入审阅用 oracle、脚本和产物，没有修改 production。桥接 C 源的已验证 SHA-256 为 `110f4e2dd15047d2071160bb573b2b5904d7469fd58ca5241c17bde07f03d2fc`。

### ABI 与完整原始 step 的数值结果

独立 [oracle.c](/D:/Projects/py/mmwcore/build/ti-gtrack-review/oracle.c) 使用自己的原生 TI 配置构造，直接调用安装来源的 `gtrack_create` / `gtrack_step`，没有调用 mmwcore bridge。对侧装 0° / 顶装 90°，分别使用无显式 variance / 显式正 variance，四种配置各跑 99 帧，共 396 帧。输入是合成点序列，不是原始 ADC 或人体研究数据。

每组覆盖动态成轨、静态点、空帧删除、第二次出生及 uid 复用；stock DETECTION(2) 与 ACTIVE(3) 都出现，uid 0 被重复使用而 tid 为 1、2。经 [compare.py](/D:/Projects/py/mmwcore/build/ti-gtrack-review/compare.py) 比较，bridge 与独立 oracle 的共同输出字段逐帧 **float32 完全相同**，包括 S9、P9×9、poststep apriori 字段、H_s、EC、gC/gD、计数器、dimension/centroid/confidence、point uid/unique/static/score、unrolled Doppler 和 presence。ABI 尺寸是 Config 296 字节、Target 1040 字节。[comparison.json](/D:/Projects/py/mmwcore/build/ti-gtrack-review/comparison.json)、[trace.jsonl](/D:/Projects/py/mmwcore/build/ti-gtrack-review/trace.jsonl)

这一检查证明上述配置与输入下的 **C 桥接保真**。它未覆盖全部配置组合、双人竞争、所有角度/速度边界，也不代替 Python/Rust API 的单元测试。benchmark 时钟值没有比较；oracle 使用原始 step 的无 benchmark 分支，bridge 使用有 benchmark 分支。

### 已复现的上游 bitmap 越界写，桥接 allocator 已规避

TI `gtrack_create.c:577/585` 按 ceil(maxNumPoints/8) 给 unique/static bitmap 分配空间；`gtrack_step.c:124/125` 按 floor(maxNumPoints/8)+1 字节清零。maxNumPoints 为 8 的倍数时，每个 bitmap 各多写 1 字节。

独立 oracle 在每次 TI 分配尾部增加 canary，记录到：容量 15 时未改变尾部；容量 16 和 1000 时，两处尾部各改变 1 字节。[probes.json](/D:/Projects/py/mmwcore/build/ti-gtrack-review/probes.json)

审阅后的 bridge 在主机 `gtrack_alloc` 每次分配尾部补 1 字节，并有源行解释。它不改 TI 数值算法；上述 396 帧零差异使用的就是该补位实现。仅给 Host 自己的 unique 数组补位不够，必须保护 TI 内部通过 allocator 创建的 bitmap。

### Poststep apriori 不是原始 Predict 快照

源码已确认 `unitUpdate` 会在静态/无动态证据分支中改写 `S_apriori_hat` 的速度、加速度，例如清零或减半；H_s 保留之前的预测测量，不同步这种更改。[unit_update:306][update]

独立实测的侧装 frame 6：poststep apriori 的径向速度已变为 0，而 H_s Doppler 仍约 0.3053 m/s。[probes.json](/D:/Projects/py/mmwcore/build/ti-gtrack-review/probes.json)

因此 `apriori_state_after_step`、`apriori_covariance_after_step` 是准确输出名称。H_s 可以称“本轮 association prediction”，但出生帧它来自 unitStart 的初始化；删除的 unit 已不在输出中。这些字段不能被宣传为对所有 track、所有阶段都完整记录的 Predict hook。若以后 RT 必须读取真正 Predict 后/Associate 前的快照，需要单独设计阶段观察机制，本次完整 stock bridge 没有该 hook。

### EC、uid/tid 与内部状态复制正确，但保留上游语义

当前 bridge 的 EC 复制官方 target EC，内部 state 从 unit 读取，point uid 保留原始特殊码，再由 Rust 将存活 uid 映射到 tid；在 oracle 序列中一致。

需要继续维持说明：EC 是本版先前 scoring 阶段保留的 inverse group covariance，不等于 post-update gC 的逆；出生时 unit 尚未经过 Score，unitStart 又没有清空所有历史字段，因此新生/复用槽上的某些字段可能是零值或上次使用留下的观测缓存。忠实报告 raw stock 字段不等于为这些字段担保“新生帧已有效”。

当 point uid 对应的 unit 在同帧 Update 中删除，原始 point association 仍可能保留 uid，而最终 target 列表没有该 uid；此时映射 `point_tid=-1` 表示无存活目标映射，不能直接说该点在 Association 阶段未被关联。

### 方差的数值边界

审阅时 Rust 接口接受非负 variance，包括全零。独立原始 TI oracle 证明：6 个相同球面位置、相同 Doppler 0.3、正 SNR 的点，配合全零四维 variance，10 帧序列会产生 NaN；普通分散点的单帧全零 variance 没有立即失败。[zero-variance-identical-oracle.txt](/D:/Projects/py/mmwcore/build/ti-gtrack-review/zero-variance-identical-oracle.txt)

这是原始 TI 在退化量测协方差下的数值边界，不是桥接偏差。建议公开安全接口要求显式 variance 严格正，或提供清楚的非有限输出失败策略；不能把非有限 state/covariance 悄悄序列化为 JSON null 后继续声称本帧有效。实际处理由生产实现与测试另行验证，本审阅没有改动算法。

### 参数、安全与分发复核范围

所查 Rust Config 已限制点/轨迹容量、整数窄化范围、有限正时间/速度/噪声相关参数、三类 box 数量与边界顺序。step 在进入 C 前拒绝超容量，不沿用 stock 的静默截断；桥接先复制输入，保护调用者数据不被 stock 的 Doppler unroll 原位修改。

C ABI 依赖 Rust 提供合法指针与足够缓冲区；size/version 校验与受控 `repr(C)` 字段顺序在当前版本匹配。独立 ctypes 对比由桥接 header 构造布局，因此上述数字一致本身不证明任意未来 ABI 都兼容，结构字段改变需要同步 ABI 版本/布局验收。

build.py 固定外部源码哈希，生成本地库、构建 manifest 和 TI-LICENSE；Rust loader 校验实际库 hash。源码与库放在安装路径及 gitignored build，未观察到本次将 TI 源码复制进可分发 Rust/Python 包。该方式清楚隔离了来源，仍须保留 TI Devices 限制，不能把本地动态库称为 Apache-only artifact。


## 来源

[project]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/examples/Industrial_and_Personal_Electronics/People_Tracking/3D_People_Tracking/src/6843/3D_people_track_6843_mss.projectspec:52
[utils]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/examples/Industrial_and_Personal_Electronics/People_Tracking/3D_People_Tracking/src/6843/mss/tracker_utils.c:274
[build]: /D:/Projects/py/openmmw/outputs/experiments/ti-allocation-pilot002-v1/build.json
[builder]: /D:/Projects/py/openmmw/outputs/experiments/ti-allocation-pilot002-v1/build_reference.py
[host]: /D:/Projects/py/openmmw/outputs/experiments/ti-allocation-pilot002-v1/host_reference.c:77
[step]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack/src/gtrack_step.c:98
[create]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack/src/gtrack_unit_create.c:109
[predict]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack/src/gtrack_unit_predict.c:83
[score]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack/src/gtrack_unit_score.c:129
[update]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack/src/gtrack_unit_update.c:87
[event]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack/src/gtrack_unit_event.c:72
[module]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack/src/gtrack_module.c:156
[report]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack/src/gtrack_unit_report.c:72
[api]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack/gtrack.h:650
[license]: /C:/ti/radar_toolbox_4_00_00_05/source/ti/custom_sdk_files/sdk3/dpu/trackerproc_overhead/packages/ti/alg/gtrack/gtrack.h:1
[cargo]: /D:/Projects/py/mmwcore/Cargo.toml:10
[current-step]: /D:/Projects/py/mmwcore/crates/mmwcore/src/tracking/measurement3d.rs:284
[current-filter]: /D:/Projects/py/mmwcore/crates/mmwcore/src/tracking/measurement3d.rs:107
[current-assoc]: /D:/Projects/py/mmwcore/crates/mmwcore/src/tracking/measurement3d.rs:307
[current-update]: /D:/Projects/py/mmwcore/crates/mmwcore/src/tracking/measurement3d.rs:369
[current-alloc]: /D:/Projects/py/mmwcore/crates/mmwcore/src/tracking/measurement3d.rs:487
[current-life]: /D:/Projects/py/mmwcore/crates/mmwcore/src/tracking/measurement3d.rs:622
[current-summary]: /D:/Projects/py/mmwcore/crates/mmwcore/src/tracking/measurement3d.rs:755
[guide]: https://dev.ti.com/tirex/explore/content/radar_toolbox_4_00_00_05/source/ti/examples/Industrial_and_Personal_Electronics/People_Tracking/3D_People_Tracking/docs/3d_people_tracking_user_guide.html
[design]: https://www.ti.com/lit/ug/tidue71c/tidue71c.pdf
[ptk]: https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/08_00_00_12/exports/docs/perception/docs/ptk_api_guide/structGTRACK__gateLimits.html
