import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import os

# 设置matplotlib为非交互模式
plt.switch_backend('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据
def load_data():
    """加载游戏周数据"""
    games = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找所有周数据文件
    weekly_files = []
    for file in os.listdir(current_dir):
        if file.endswith('_weekly_data.csv'):
            weekly_files.append(file)
    
    if not weekly_files:
        print("警告：未找到task1_2.py生成的周数据文件")
        return {}
    
    print(f"找到 {len(weekly_files)} 个游戏周数据文件")
    
    for file in weekly_files:
        try:
            # 从文件名提取游戏名称
            game_name = file.replace('_weekly_data.csv', '').replace('_', ' ')
            
            # 读取CSV文件
            df = pd.read_csv(os.path.join(current_dir, file), index_col=0, parse_dates=True)
            
            games[game_name] = df
            print(f"  成功加载 {game_name}: {len(df)} 条记录")
            
        except Exception as e:
            print(f"  加载文件 {file} 失败: {e}")
    
    return games

# 构建模型数据
def prepare_model_data(df):
    """准备模型数据"""
    # 提取特征和目标变量
    # V(t+1) = αV(t) + η·V(t)^δ + Σ(βk·D(t-k)) + Σ(γk·U(t-k)) + ε(t)
    
    # 假设：
    # D(t) = 0（无打折数据）
    # U(t) 包括：positive_rate（好评率）、total_reviews（评论数）、avg_playtime（平均游玩时长）
    # L = 1（滞后1周）
    
    X = []
    y = []
    
    # 处理缺失值
    df = df.dropna()
    
    # 需要至少2周的数据来计算滞后项
    for i in range(1, len(df) - 1):
        # 当前时间t的特征
        V_t = df['V'].iloc[i]
        
        # 滞后1周的特征（t-1）
        positive_rate_lag1 = df['positive_rate'].iloc[i-1]
        total_reviews_lag1 = df['total_reviews'].iloc[i-1]
        avg_playtime_lag1 = df['avg_playtime'].iloc[i-1]
        
        # 下一时间t+1的热度
        V_t1 = df['V'].iloc[i+1]
        
        # 检查是否有NaN值
        if not np.isnan(V_t) and not np.isnan(positive_rate_lag1) and not np.isnan(total_reviews_lag1) and not np.isnan(avg_playtime_lag1) and not np.isnan(V_t1):
            # 构建特征向量
            X.append([V_t, positive_rate_lag1, total_reviews_lag1, avg_playtime_lag1])
            y.append(V_t1)
    
    return np.array(X), np.array(y)

# 非线性模型函数
def nonlinear_model(x, alpha, eta, delta, beta0, gamma1, gamma2, gamma3, intercept):
    """非线性模型函数"""
    V_t = x[:, 0]
    positive_rate = x[:, 1]
    total_reviews = x[:, 2]
    avg_playtime = x[:, 3]
    
    # V(t+1) = αV(t) + η·V(t)^δ + β0·D(t) + γ1·positive_rate + γ2·total_reviews + γ3·avg_playtime + intercept
    # 由于D(t)=0，所以省略
    return alpha * V_t + eta * (V_t ** delta) + gamma1 * positive_rate + gamma2 * total_reviews + gamma3 * avg_playtime + intercept

# 自定义目标函数
def custom_objective(params, X, y, time_weights=None):
    """
    自定义目标函数
    参数:
    - params: 模型参数 [alpha, eta, delta, beta0, gamma1, gamma2, gamma3, intercept]
    - X: 特征矩阵
    - y: 目标值
    - time_weights: 时间权重向量
    
    返回:
    - 目标函数值（越小越好）
    """
    # 计算预测值
    y_pred = nonlinear_model(X, *params)
    
    # 计算误差
    errors = y - y_pred
    
    # 基础均方误差
    mse = np.mean(errors**2)
    
    # 如果提供了时间权重，计算加权MSE
    if time_weights is not None:
        weighted_errors = errors * time_weights
        mse = np.mean(weighted_errors**2)
    
    # 添加惩罚项
    # 1. 惩罚过大的参数值
    param_penalty = np.sum(np.abs(params[:3]) * 0.001)  # 对前三个参数的惩罚
    
    # 2. 惩罚非线性衰减指数不在合理范围内
    delta = params[2]
    delta_penalty = 0
    if delta < 1:  # 希望delta > 1
        delta_penalty = (1 - delta) * 0.1
    
    # 3. 惩罚非线性衰减强度为正（希望eta < 0）
    eta = params[1]
    eta_penalty = 0
    if eta > 0:
        eta_penalty = eta * 0.1
    
    # 总目标函数值
    total_error = mse + param_penalty + delta_penalty + eta_penalty
    
    return total_error

# 模型参数寻优
def optimize_parameters(X, y):
    """使用非线性模型进行参数寻优"""
    from scipy.optimize import minimize
    
    # 初始参数猜测
    initial_guess = [0.5, -0.1, 2.0, 0, 0.1, 0.001, 0.001, 0]
    
    # 生成时间权重（越近的时间权重越大）
    n = len(y)
    time_weights = np.linspace(0.5, 1.5, n)  # 权重从0.5线性增加到1.5
    time_weights /= np.sum(time_weights)  # 归一化
    
    # 定义优化目标函数（无约束）
    def objective(params):
        return custom_objective(params, X, y, time_weights)
    
    # 定义有约束的优化目标函数
    def objective_with_constraints(params):
        return custom_objective(params, X, y, time_weights)
    
    # 约束条件
    constraints = [
        {'type': 'ineq', 'fun': lambda params: -params[1]},  # eta < 0
        {'type': 'ineq', 'fun': lambda params: params[2] - 1}  # delta > 1
    ]
    
    # 边界条件
    bounds = [
        (0, 1),      # alpha: 0 < alpha < 1（模型稳定）
        (-10, 0),    # eta: eta < 0
        (1, 10),     # delta: delta > 1
        (0, 0),      # beta0: 固定为0（无折扣数据）
        (-1, 1),     # gamma1: 好评率系数
        (-0.01, 0.01), # gamma2: 评论数系数
        (-0.01, 0.01), # gamma3: 平均游玩时长系数
        (-1, 1)      # intercept: 截距
    ]
    
    try:
        # 先尝试有约束的优化
        print("  执行有约束的参数寻优...")
        result = minimize(objective_with_constraints, initial_guess, 
                         method='SLSQP', bounds=bounds, constraints=constraints, 
                         options={'maxiter': 1000, 'disp': False})
        
        if not result.success:
            # 如果有约束优化失败，尝试无约束优化
            print("  有约束优化失败，尝试无约束优化...")
            result = minimize(objective, initial_guess, method='BFGS', 
                             options={'maxiter': 1000, 'disp': False})
        
        popt = result.x
        
        # 获取参数
        alpha = popt[0]  # 线性衰减系数
        eta = popt[1]    # 非线性衰减强度
        delta = popt[2]   # 非线性衰减指数
        beta0 = popt[3]   # 折扣系数（当前为0）
        gamma1 = popt[4]  # 好评率系数
        gamma2 = popt[5]  # 评论数系数
        gamma3 = popt[6]  # 平均游玩时长系数
        intercept = popt[7]  # 截距
        
        # 预测
        y_pred = nonlinear_model(X, *popt)
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # 计算自定义目标函数值
        custom_error = custom_objective(popt, X, y, time_weights)
        
        return {
            'params': popt,
            'alpha': alpha,
            'eta': eta,
            'delta': delta,
            'beta0': beta0,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'gamma3': gamma3,
            'intercept': intercept,
            'mse': mse,
            'r2': r2,
            'custom_error': custom_error,
            'y_pred': y_pred
        }
    except Exception as e:
        print(f"  拟合模型失败: {e}")
        # 返回默认参数
        y_pred = nonlinear_model(X, *initial_guess)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        custom_error = custom_objective(initial_guess, X, y, time_weights)
        
        return {
            'params': initial_guess,
            'alpha': initial_guess[0],
            'eta': initial_guess[1],
            'delta': initial_guess[2],
            'beta0': initial_guess[3],
            'gamma1': initial_guess[4],
            'gamma2': initial_guess[5],
            'gamma3': initial_guess[6],
            'intercept': initial_guess[7],
            'mse': mse,
            'r2': r2,
            'custom_error': custom_error,
            'y_pred': y_pred
        }

# 可视化拟合效果
def plot_fitting(y_true, y_pred, game_name):
    """可视化拟合效果"""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_true)), y_true, label='实际热度', linewidth=2)
    plt.plot(range(len(y_pred)), y_pred, label='预测热度', linestyle='--', linewidth=2)
    plt.title(f'{game_name} 热度预测拟合效果')
    plt.xlabel('时间（周）')
    plt.ylabel('热度指数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{game_name}_fitting.png')
    plt.close()
    print(f"{game_name} 拟合效果图已保存为 {game_name}_fitting.png")

# 主函数
def main():
    print("开始执行Task 2.2：参数寻优")
    
    # 加载数据
    games = load_data()
    
    if not games:
        print("\n错误：未能加载任何游戏数据，请确保已运行task1_2.py")
        return
    
    for game_name, df in games.items():
        print(f"\n=== {game_name} 参数寻优 ===")
        
        # 准备模型数据
        X, y = prepare_model_data(df)
        
        if len(X) < 10:
            print(f"  数据不足，跳过参数寻优")
            continue
        
        # 参数寻优
        result = optimize_parameters(X, y)
        
        # 打印参数
        print("模型参数:")
        print(f"α (线性衰减系数): {result['alpha']:.4f}")
        print(f"η (非线性衰减强度): {result['eta']:.4f}")
        print(f"δ (非线性衰减指数): {result['delta']:.4f}")
        print(f"β0 (折扣系数): {result['beta0']:.4f}")
        print(f"γ1 (好评率系数): {result['gamma1']:.4f}")
        print(f"γ2 (评论数系数): {result['gamma2']:.4f}")
        print(f"γ3 (平均游玩时长系数): {result['gamma3']:.4f}")
        print(f"截距: {result['intercept']:.4f}")
        
        # 打印评估指标
        print("\n评估指标:")
        print(f"均方误差 (MSE): {result['mse']:.4f}")
        print(f"R2 评分: {result['r2']:.4f}")
        print(f"自定义目标函数值: {result['custom_error']:.4f}")
        
        # 可视化拟合效果
        plot_fitting(y, result['y_pred'], game_name)
    
    print("\nTask 2.2 完成！")

if __name__ == "__main__":
    main()
