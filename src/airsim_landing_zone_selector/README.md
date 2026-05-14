# AirSim 深度图安全降落点选择模块

本项目使用 AirSim 深度相机网格数据，评估候选降落区域的净空、平整度和坡度，自动选择安全降落点。

## 主要内容
- 读取 AirSim 深度图网格数据。
- 使用滑动窗口评估候选降落区域。
- 计算 clearance、flatness、slope 和 landing_score。
- 输出 best / safe / reject 三类决策。
- 生成深度网格选点图和候选区域得分排序图。

## 运行
```bash
python src/airsim_landing_zone_selector/landing_zone.py --output docs/pr_assets/airsim_landing_zone_selector
python src/airsim_landing_zone_selector/tests/test_landing_zone.py
```
