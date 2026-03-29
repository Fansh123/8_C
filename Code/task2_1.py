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

# 模型理论分析与推导
def model_analysis():
    """模型理论分析与推导"""
    print("=== Task 2.1 模型构建：理论分析与推导 ===")
    print("\n1. 模型结构")
    print("   V(t+1) = αV(t) + η·V(t)^δ + Σ(βk·D(t-k)) + Σ(γk·U(t-k)) + ε(t)")
    print("   其中：")
    print("   - V(t)：t时刻的游戏热度")
    print("   - V(t+1)：t+1时刻的游戏热度")
    print("   - α：线性衰减系数")
    print("   - η：非线性衰减强度")
    print("   - δ：非线性衰减的指数")
    print("   - D(t-k)：第k周前的折扣力度")
    print("   - U(t-k)：第k周前的其他影响因素")
    print("   - βk：第k周前折扣的边际贡献")
    print("   - γk：第k周前其他因素的边际贡献")
    print("   - L：最大滞后周数")
    print("   - ε(t)：随机误差项")
    
    print("\n2. 模型假设")
    print("   (1) 热度具有自回归特性，包含线性和非线性成分")
    print("   (2) 折扣和其他因素对热度的影响具有滞后效应")
    print("   (3) 非线性衰减项通常为负值，表示加速衰减")
    print("   (4) 误差项ε(t)服从均值为0的正态分布")
    
    print("\n3. 模型理论推导")
    print("   (1) 线性自回归项αV(t)：")
    print("      - α表示热度的线性持续性，0 < α < 1时，热度会逐渐衰减")
    
    print("   (2) 非线性自回归项η·V(t)^δ：")
    print("      - η通常为负值，表示加速衰减")
    print("      - δ > 1，使早期衰减快，后期慢")
    
    print("   (3) 滞后效应项：")
    print("      - Σ(βk·D(t-k))：折扣的滞后影响")
    print("      - Σ(γk·U(t-k))：其他因素的滞后影响")
    
    print("   (4) 误差项ε(t)：")
    print("      - 捕获模型未考虑的随机因素")
    print("      - 假设ε(t) 服从均值为0的正态分布")
    
    print("\n4. 模型意义")
    print("   - 该模型是一个非线性动态系统模型，更准确地描述了游戏热度随时间的演变规律")
    print("   - 考虑了热度的非线性衰减特性")
    print("   - 考虑了折扣和其他因素的滞后效应")
    print("   - 可以更准确地预测未来热度，为游戏运营策略提供参考")

# 构建模型
def build_model(df):
    """构建热度演变模型"""
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

# 训练模型
def train_model(X, y):
    """训练模型"""
    # 初始参数猜测
    initial_guess = [0.5, -0.1, 2.0, 0, 0.1, 0.001, 0.001, 0]
    
    # 拟合非线性模型
    try:
        popt, pcov = curve_fit(nonlinear_model, X, y, p0=initial_guess, maxfev=10000)
        
        # 获取参数
        alpha = popt[0]  # 线性衰减系数
        eta = popt[1]    # 非线性衰减强度
        delta = popt[2]   # 非线性衰减指数
        beta0 = popt[3]   # 折扣系数（当前为0）
        gamma1 = popt[4]  # 好评率系数
        gamma2 = popt[5]  # 评论数系数
        gamma3 = popt[6]  # 平均游玩时长系数
        intercept = popt[7]  # 截距
        
        return {
            'params': popt,
            'alpha': alpha,
            'eta': eta,
            'delta': delta,
            'beta0': beta0,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'gamma3': gamma3,
            'intercept': intercept
        }
    except Exception as e:
        print(f"拟合模型失败: {e}")
        # 返回默认参数
        return {
            'params': initial_guess,
            'alpha': initial_guess[0],
            'eta': initial_guess[1],
            'delta': initial_guess[2],
            'beta0': initial_guess[3],
            'gamma1': initial_guess[4],
            'gamma2': initial_guess[5],
            'gamma3': initial_guess[6],
            'intercept': initial_guess[7]
        }

# 测试模型
def test_model(model, X, y):
    """测试模型"""
    # 预测
    y_pred = nonlinear_model(X, *model['params'])
    
    # 计算评估指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        'y_pred': y_pred,
        'mse': mse,
        'r2': r2
    }

# 可视化模型预测结果
def plot_prediction(y_true, y_pred, game_name):
    """可视化模型预测结果"""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_true)), y_true, label='实际热度', linewidth=2)
    plt.plot(range(len(y_pred)), y_pred, label='预测热度', linestyle='--', linewidth=2)
    plt.title(f'{game_name} 热度预测结果')
    plt.xlabel('时间（周）')
    plt.ylabel('热度指数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{game_name}_model_prediction.png')
    plt.close()
    print(f"{game_name} 模型预测结果图已保存为 {game_name}_model_prediction.png")

# 模型稳定性分析
def stability_analysis(alpha):
    """分析模型稳定性"""
    print(f"\n模型稳定性分析：")
    print(f"自回归系数 a = {alpha:.4f}")
    
    if alpha < 0:
        print("模型不稳定：a为负值，热度可能出现振荡")
    elif alpha >= 1:
        print("模型不稳定：a≥1，热度可能无限增长")
    else:
        print("模型稳定：a在(0,1)之间，热度会逐渐收敛")
        print(f"热度收敛速度：每周期衰减 {100*(1-alpha):.2f}%")

# 主函数
def main():
    print("开始执行Task 2.1：模型构建")
    
    # 1. 模型理论分析与推导
    model_analysis()
    
    # 2. 加载数据
    games = load_data()
    
    if not games:
        print("\n错误：未能加载任何游戏数据，请确保已运行task1_2.py")
        return
    
    # 3. 对每个游戏构建和测试模型
    for game_name, df in games.items():
        print(f"\n=== {game_name} 模型构建与测试 ===")
        
        # 准备模型数据
        X, y = build_model(df)
        
        if len(X) < 10:
            print(f"  数据不足，跳过模型构建")
            continue
        
        # 训练模型
        model_result = train_model(X, y)
        
        # 测试模型
        test_result = test_model(model_result, X, y)
        
        # 打印模型参数
        print("模型参数:")
        print(f"α (线性衰减系数): {model_result['alpha']:.4f}")
        print(f"η (非线性衰减强度): {model_result['eta']:.4f}")
        print(f"δ (非线性衰减指数): {model_result['delta']:.4f}")
        print(f"β0 (折扣系数): {model_result['beta0']:.4f}")
        print(f"γ1 (好评率系数): {model_result['gamma1']:.4f}")
        print(f"γ2 (评论数系数): {model_result['gamma2']:.4f}")
        print(f"γ3 (平均游玩时长系数): {model_result['gamma3']:.4f}")
        print(f"截距: {model_result['intercept']:.4f}")
        
        # 打印模型评估指标
        print("\n模型评估:")
        print(f"均方误差 (MSE): {test_result['mse']:.4f}")
        print(f"R2 评分: {test_result['r2']:.4f}")
        
        # 分析模型稳定性
        stability_analysis(model_result['alpha'])
        
        # 可视化预测结果
        plot_prediction(y, test_result['y_pred'], game_name)
    
    print("\nTask 2.1 模型构建完成！")

if __name__ == "__main__":
    main()
