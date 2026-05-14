# MuJoCo 足底接触力平衡分析模块

本项目使用 MuJoCo MJCF 模型、`cfrc_ext` 接触力导出日志和 `sensordata` 质心位置记录，分析左右支撑力差、质心偏移和躯干横滚角对机器人稳定性的影响，并生成姿态恢复建议。

## 主要内容
- 读取 MuJoCo MJCF 模型 `mujoco_quadruped_balance.xml`。
- 读取 MuJoCo `cfrc_ext` 足底接触力和 `sensordata` 质心状态数据。
- 计算左右支撑力不平衡、质心偏移和综合平衡风险。
- 输出 stable_walk / adjust_support / recover_posture 三类动作建议。
- 生成平衡风险曲线、左右接触力分布图和支撑多边形/质心投影回放图。

## 运行
```bash
python src/mujoco_contact_balance_analyzer/contact_balance.py --output docs/pr_assets/mujoco_contact_balance_analyzer
python src/mujoco_contact_balance_analyzer/tests/test_contact_balance.py
```
