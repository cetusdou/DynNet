import torch

def euler_solver_sde(model, y0, num_steps=5000, num_genes=7):
    state = y0
    trajectory = [state.clone()]  # 初始化轨迹列表
    num_trajectories = y0.shape[0]
    for i in range(num_steps):
        _, state = model(state)
        # 设置吸收边界条件,将小于0的值设为0
        state = torch.clamp(state, min=0)
        trajectory.append(state.clone())  # 保存每一步的状态,保留梯度信息
    return trajectory