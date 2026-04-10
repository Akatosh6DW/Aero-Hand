"""
RMA (Rapid Motor Adaptation) 师生蒸馏参考实现
==============================================================================
用途：为 Aero Hand 灵巧手的 Sim2Real 触觉跨越提供核心网络架构参考。
框架：PyTorch（参考实现），实际部署可移植至 JAX/Flax。

核心思想：
  Phase 1 (Teacher): 在仿真中用精确的多维接触力训练 Teacher 编码器，
                     将 3D 指尖力编码为低维隐变量 z_t。
  Phase 2 (Student): 训练 Adaptation Module，仅用带噪声的本体感觉历史序列
                     预测 z_t，部署时不依赖精确力传感器。

参考文献：
  [1] Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots", RSS 2021
  [2] Qi et al., "In-Hand Object Rotation via Rapid Motor Adaptation", CoRL 2022
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================================
#  1. Teacher 环境编码器 (Environment Encoder)
# ============================================================================
class TeacherEnvironmentEncoder(nn.Module):
    """将仿真中精确的多维指尖接触力编码为低维隐变量。

    输入：MuJoCo efc_force 提取的 5 指 × 3D 接触力 = 15 维
          （或 5 指 × 1D 法向力 = 5 维，取决于传感器配置）
    输出：低维隐变量 z_t ∈ R^{latent_dim}

    网络结构:
      force_input (15D) → MLP(128, 64) → z_t (8D)

    设计考量：
      - 15D→8D 的压缩比约 2:1，保留力方向和大小的关键信息
      - 2 层 MLP 足够（接触力本身已是较低层特征）
      - 使用 ELU 激活（比 ReLU 在负区间有非零梯度，适合力信号）
      - 最终层不加激活，允许隐变量取任意实值
    """

    def __init__(
        self,
        force_dim: int = 15,     # 5根手指 × 3D力（法向+切向）
        hidden_dim: int = 128,   # 隐藏层宽度
        latent_dim: int = 8,     # 隐变量维度（经验值：6~16）
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(force_dim, hidden_dim),
            nn.ELU(),                           # ELU: 负区间有梯度，适合力信号
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, latent_dim),  # 无激活：z_t 是无约束隐变量
        )

    def forward(self, force_3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            force_3d: (batch, force_dim) 精确的 3D 指尖接触力
        Returns:
            z_t: (batch, latent_dim) 环境隐变量
        """
        return self.encoder(force_3d)


# ============================================================================
#  2. Student 适应模块 (Adaptation Module)
# ============================================================================
class StudentAdaptationModule(nn.Module):
    """用本体感觉历史序列预测 Teacher 的隐变量 z_t。

    部署时无法获取精确 3D 接触力，改用过去 H 帧的本体感觉信息
    （关节角度 + 关节速度 + 电机目标 + 带噪声的法向力估计）来预测 z_t。

    架构选项：
      - MLP: 将 H 帧 concat 成一个长向量，简单高效
      - TCN: 时序卷积，适合捕捉力的动态变化模式

    此处提供 TCN 实现（推荐），可换为 MLP。
    """

    def __init__(
        self,
        proprio_dim: int = 17,    # 每帧本体感觉维度 (motor_targets=6 + tactile=5 + last_act=6)
        history_len: int = 50,    # 历史窗口长度 H（50帧 × 0.05s = 2.5s）
        latent_dim: int = 8,      # 需匹配 Teacher 的 latent_dim
        tcn_channels: list = None,  # TCN 各层通道数
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [32, 64, 64]

        # ── TCN (Temporal Convolution Network) ──────────────────────────────
        # 因果卷积：只看过去，不看未来（causal padding）
        layers = []
        in_channels = proprio_dim
        for out_channels in tcn_channels:
            layers.append(
                CausalConv1d(in_channels, out_channels, kernel_size=3, dilation=1)
            )
            layers.append(nn.ELU())
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)

        # ── 全局池化 + 映射到隐变量 ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(tcn_channels[-1], 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (batch, H, proprio_dim) 过去 H 帧的本体感觉序列
        Returns:
            z_hat: (batch, latent_dim) 预测的环境隐变量
        """
        # TCN 输入格式: (batch, channels, seq_len)
        x = history.transpose(1, 2)       # (batch, proprio_dim, H)
        x = self.tcn(x)                   # (batch, tcn_channels[-1], H)
        # 全局平均池化：取所有时间步的均值
        x = x.mean(dim=-1)               # (batch, tcn_channels[-1])
        return self.head(x)               # (batch, latent_dim)


class CausalConv1d(nn.Module):
    """因果一维卷积：左 padding 确保只看过去。"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation,
        )

    def forward(self, x):
        # 左填零，确保因果性
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


# ============================================================================
#  3. 完整的 RMA Actor（集成 Teacher/Student + Base Policy）
# ============================================================================
class RMAActor(nn.Module):
    """RMA 策略网络：base_policy(proprio, z_t) → action。

    训练流程：
      Phase 1: 冻结 adaptation_module，用 env_encoder 提供 z_t 训练 base_policy
               目标：学会在已知环境参数下最优控制
      Phase 2: 冻结 base_policy + env_encoder，训练 adaptation_module
               通过蒸馏 z_hat → z_t 来学习从历史推断环境
      Phase 3 (可选): 端到端微调所有模块
    """

    def __init__(
        self,
        proprio_dim: int = 17,
        force_dim: int = 15,
        action_dim: int = 6,
        latent_dim: int = 8,
        history_len: int = 50,
        policy_hidden: Tuple[int, ...] = (512, 256, 128),
    ):
        super().__init__()
        self.env_encoder = TeacherEnvironmentEncoder(
            force_dim=force_dim, latent_dim=latent_dim,
        )
        self.adaptation_module = StudentAdaptationModule(
            proprio_dim=proprio_dim, history_len=history_len, latent_dim=latent_dim,
        )

        # ── Base Policy MLP ─────────────────────────────────────────────────
        # 输入：当前 proprio (17D) + 隐变量 z_t (8D) = 25D
        layers = []
        in_dim = proprio_dim + latent_dim
        for h in policy_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())    # 与当前 Brax PPO 一致
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())        # 动作范围 [-1, 1]
        self.base_policy = nn.Sequential(*layers)

    def forward_teacher(self, proprio: torch.Tensor, force_3d: torch.Tensor) -> torch.Tensor:
        """Phase 1 前向：使用精确力（仿真训练时）。"""
        z_t = self.env_encoder(force_3d)
        return self.base_policy(torch.cat([proprio, z_t], dim=-1)), z_t

    def forward_student(self, proprio: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Phase 2/部署 前向：使用历史序列预测隐变量。"""
        z_hat = self.adaptation_module(history)
        return self.base_policy(torch.cat([proprio, z_hat], dim=-1)), z_hat


# ============================================================================
#  4. 蒸馏损失 (Distillation Loss)
# ============================================================================
class RMADistillationLoss(nn.Module):
    """计算 Teacher z_t 与 Student z_hat 之间的蒸馏损失。

    在 PPO 框架下的集成方式：

    方案 A（分阶段训练，推荐）：
      Phase 1: 正常 PPO 训练 base_policy + env_encoder
               L = L_policy + L_value （标准 PPO loss）
      Phase 2: 冻结 policy 和 encoder，只训练 adaptation_module
               L = MSE(z_hat, z_t.detach())
      优点：训练稳定，各阶段目标清晰
      缺点：需要两阶段训练

    方案 B（联合训练）：
      L = L_policy + L_value + β * MSE(z_hat, z_t.detach())
      β 从 0 线性增长到 0.5（课程学习）
      优点：端到端一次完成
      缺点：超参数多，可能不稳定
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta   # 蒸馏损失权重

    def forward(
        self,
        z_teacher: torch.Tensor,
        z_student: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_teacher: (batch, latent_dim) Teacher 编码的隐变量
            z_student: (batch, latent_dim) Student 预测的隐变量
        Returns:
            loss: 标量，MSE 蒸馏损失
        """
        # 关键：z_teacher.detach() 确保不会传梯度回 Teacher
        loss = F.mse_loss(z_student, z_teacher.detach())
        return self.beta * loss


# ============================================================================
#  5. Reward Shaping 参考代码（已集成到 JAX 环境中）
# ============================================================================
def reward_force_overload_nonlinear(
    tip_force: torch.Tensor,
    f_max: float = 2.8,
    soft_width: float = 0.5,
) -> torch.Tensor:
    """非线性过载惩罚：quadratic + cubic 平滑过渡。

    ┌──────────────────────────────────────────┐
    │  cost                                     │
    │   ╱                                       │
    │  ╱ cubic 主导                             │
    │ ╱                                         │
    │╱ quadratic 主导                           │
    │──────────┬─────────────── |f|            │
    │        F_max (2.8N)                       │
    └──────────────────────────────────────────┘

    当 |f| < F_max: cost = 0
    当 |f| > F_max: cost = overload^2 + (overload/W)^3

    权重推荐：scale = -1.0 ~ -2.0
    """
    abs_f = tip_force.abs()
    overload = (abs_f - f_max).clamp(min=0)
    cost = overload ** 2 + (overload / (soft_width + 1e-6)) ** 3
    return cost.mean()


def reward_soft_contact(
    tip_force: torch.Tensor,
    f_min: float = 0.1,
    f_max: float = 2.5,
) -> torch.Tensor:
    """安全区间内的 bell-shaped 有效接触奖励。

    ┌──────────────────────────────────────────┐
    │  reward                                   │
    │         ┌──────────────┐                  │
    │        ╱                ╲                 │
    │       ╱                  ╲                │
    │──────┼──────────────────┼── |f|          │
    │    F_min (0.1N)     F_max (2.5N)         │
    └──────────────────────────────────────────┘

    用 sigmoid 平滑过渡：
      lower_gate = σ(20 * (|f| - F_min))  -- 左侧上升
      upper_gate = σ(10 * (F_max - |f|))  -- 右侧下降

    权重推荐：scale = 3.0 ~ 6.0
    MuJoCo 量级参考：力和扭矩惩罚建议 0.001~0.01 量级（相对奖励~100~200），
                     但在我们的系统中奖励已归一化为 0~1，故 scale=3~6 合适。
    """
    abs_f = tip_force.abs()
    lower_gate = torch.sigmoid(20.0 * (abs_f - f_min))
    upper_gate = torch.sigmoid(10.0 * (f_max - abs_f))
    return (lower_gate * upper_gate).mean()


# ============================================================================
#  6. 使用示例
# ============================================================================
if __name__ == "__main__":
    batch_size = 256
    proprio_dim = 17   # motor_targets(6) + tactile(5) + last_act(6)
    force_dim = 15     # 5 fingers × 3D force
    action_dim = 6     # 6 通道控制
    history_len = 50   # 2.5s @ 20Hz
    latent_dim = 8

    # 创建模型
    model = RMAActor(
        proprio_dim=proprio_dim,
        force_dim=force_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        history_len=history_len,
    )

    # 模拟数据
    proprio = torch.randn(batch_size, proprio_dim)
    force_3d = torch.randn(batch_size, force_dim).abs()  # 力为正值
    history = torch.randn(batch_size, history_len, proprio_dim)

    # Phase 1: Teacher 前向
    action_t, z_t = model.forward_teacher(proprio, force_3d)
    print(f"Teacher action: {action_t.shape}, z_t: {z_t.shape}")

    # Phase 2: Student 前向
    action_s, z_hat = model.forward_student(proprio, history)
    print(f"Student action: {action_s.shape}, z_hat: {z_hat.shape}")

    # 蒸馏损失
    distill_loss_fn = RMADistillationLoss(beta=1.0)
    loss = distill_loss_fn(z_t, z_hat)
    print(f"Distillation loss: {loss.item():.4f}")

    # Reward shaping 示例
    tip_f = torch.tensor([0.05, 0.3, 1.5, 2.8, 3.5])
    print(f"\n--- Reward Shaping Demo ---")
    print(f"Tip forces:       {tip_f.tolist()}")
    print(f"Overload penalty:  {reward_force_overload_nonlinear(tip_f):.4f}")
    print(f"Soft contact:      {reward_soft_contact(tip_f):.4f}")

    # 参数量统计
    total = sum(p.numel() for p in model.parameters())
    encoder_p = sum(p.numel() for p in model.env_encoder.parameters())
    adapt_p = sum(p.numel() for p in model.adaptation_module.parameters())
    policy_p = sum(p.numel() for p in model.base_policy.parameters())
    print(f"\n--- Model Parameters ---")
    print(f"Total:       {total:>8,d}")
    print(f"Env Encoder: {encoder_p:>8,d}")
    print(f"Adaptation:  {adapt_p:>8,d}")
    print(f"Base Policy: {policy_p:>8,d}")
