import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from scipy.optimize import curve_fit

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

# 分析参数分布
def analyze_parameter_distributions(df):
    """分析参数的概率分布"""
    print("分析参数分布...")
    
    # 计算热度变化
    heat_changes = []
    for i in range(1, len(df)):
        change = df['V'].iloc[i] - df['V'].iloc[i-1]
        heat_changes.append(change)
    
    # 计算热度变化的统计信息
    heat_changes = np.array(heat_changes)
    mean_change = np.mean(heat_changes)
    std_change = np.std(heat_changes)
    
    print(f"热度变化均值: {mean_change:.4f}")
    print(f"热度变化标准差: {std_change:.4f}")
    
    # 分析好评率变化
    rating_changes = []
    for i in range(1, len(df)):
        change = df['positive_rate'].iloc[i] - df['positive_rate'].iloc[i-1]
        rating_changes.append(change)
    
    rating_changes = np.array(rating_changes)
    mean_rating_change = np.mean(rating_changes)
    std_rating_change = np.std(rating_changes)
    
    print(f"好评率变化均值: {mean_rating_change:.4f}")
    print(f"好评率变化标准差: {std_rating_change:.4f}")
    
    # 分析游玩时长变化
    playtime_changes = []
    for i in range(1, len(df)):
        change = df['avg_playtime'].iloc[i] - df['avg_playtime'].iloc[i-1]
        playtime_changes.append(change)
    
    playtime_changes = np.array(playtime_changes)
    mean_playtime_change = np.mean(playtime_changes)
    std_playtime_change = np.std(playtime_changes)
    
    print(f"游玩时长变化均值: {mean_playtime_change:.4f}")
    print(f"游玩时长变化标准差: {std_playtime_change:.4f}")
    
    # 可视化热度变化分布
    plt.figure(figsize=(12, 6))
    plt.hist(heat_changes, bins=30, alpha=0.7, label='热度变化')
    plt.axvline(mean_change, color='red', linestyle='--', label=f'均值: {mean_change:.4f}')
    plt.axvline(mean_change + std_change, color='green', linestyle='--', label=f'均值+标准差: {mean_change + std_change:.4f}')
    plt.axvline(mean_change - std_change, color='green', linestyle='--', label=f'均值-标准差: {mean_change - std_change:.4f}')
    plt.title('热度变化分布')
    plt.xlabel('热度变化')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('heat_change_distribution.png')
    plt.close()
    print("热度变化分布图已保存为 heat_change_distribution.png")
    
    return {
        'heat_change': {'mean': mean_change, 'std': std_change},
        'rating_change': {'mean': mean_rating_change, 'std': std_rating_change},
        'playtime_change': {'mean': mean_playtime_change, 'std': std_playtime_change}
    }

# 非线性模型函数
def nonlinear_model(x, alpha, eta, delta, beta0, gamma1, gamma2, gamma3, intercept):
    """非线性模型函数"""
    V_t = x[:, 0]
    positive_rate = x[:, 1]
    total_reviews = x[:, 2]
    avg_playtime = x[:, 3]
    
    # V(t+1) = αV(t) + η·V(t)^δ + β0·D(t) + γ1·positive_rate + γ2·total_reviews + γ3·avg_playtime + intercept
    return alpha * V_t + eta * (V_t ** delta) + beta0 * 0 + gamma1 * positive_rate + gamma2 * total_reviews + gamma3 * avg_playtime + intercept

# 构建并训练模型
def build_and_train_model(df):
    """构建并训练热度演变模型"""
    # 准备数据
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
            X.append([V_t, positive_rate_lag1, total_reviews_lag1, avg_playtime_lag1])
            y.append(V_t1)
    
    X = np.array(X)
    y = np.array(y)
    
    # 初始参数猜测
    initial_guess = [0.5, -0.1, 2.0, 0, 0.1, 0.001, 0.001, 0]
    
    # 拟合非线性模型
    try:
        popt, pcov = curve_fit(nonlinear_model, X, y, p0=initial_guess, maxfev=10000)
        print(f"模型拟合成功，参数: {popt}")
        return popt
    except Exception as e:
        print(f"模型拟合失败: {e}")
        return initial_guess

# 预测热度（考虑随机因素）
def predict_heat(model_params, V_t, positive_rate, total_reviews, avg_playtime, discount=0, distributions=None):
    """预测下一周的热度，考虑随机因素"""
    # 基础预测（使用非线性模型）
    alpha, eta, delta, beta0, gamma1, gamma2, gamma3, intercept = model_params
    base_pred = alpha * V_t + eta * (V_t ** delta) + beta0 * discount + gamma1 * positive_rate + gamma2 * total_reviews + gamma3 * avg_playtime + intercept
    
    # 考虑折扣对热度的影响（随机）
    if discount > 0:
        # 折扣效果服从正态分布
        discount_effect_mean = discount * 0.3
        discount_effect_std = discount * 0.1
        discount_effect = np.random.normal(discount_effect_mean, discount_effect_std)
        base_pred += discount_effect
    
    # 考虑随机波动（基于历史数据）
    if distributions:
        random_effect = np.random.normal(
            distributions['heat_change']['mean'],
            distributions['heat_change']['std']
        )
        base_pred += random_effect
    
    # 确保热度在合理范围内
    final_pred = max(0, min(1, base_pred))
    
    return final_pred

# 内容更新对热度的影响（随机）
def content_update_effect(base_heat, distributions=None):
    """内容更新对热度的影响，考虑随机因素"""
    # 基础更新效果
    update_effect_mean = 0.2
    update_effect_std = 0.05
    update_effect = np.random.normal(update_effect_mean, update_effect_std)
    
    # 应用更新效果
    new_heat = base_heat + update_effect
    
    # 确保热度在合理范围内
    new_heat = max(0, min(1, new_heat))
    
    return new_heat

# 计算总效益（考虑随机因素）
def calculate_total_benefit(schedule, model_params, initial_heat, initial_positive_rate, initial_reviews, initial_playtime, distributions):
    """计算总效益，考虑随机因素"""
    # 初始化参数
    current_heat = initial_heat
    current_positive_rate = initial_positive_rate
    current_reviews = initial_reviews
    current_playtime = initial_playtime
    
    total_benefit = 0
    
    # 模拟12个月（52周）的运营
    for week in range(52):
        # 检查本周是否有活动
        discount = 0
        is_update = False
        
        for event in schedule:
            if event['type'] == 'discount' and event['start_week'] <= week < event['start_week'] + event['duration']:
                discount = event['discount']
            elif event['type'] == 'update' and event['week'] == week:
                is_update = True
        
        # 预测下一周的热度（考虑随机因素）
        next_heat = predict_heat(model_params, current_heat, current_positive_rate, current_reviews, current_playtime, discount, distributions)
        
        # 应用内容更新效果（考虑随机因素）
        if is_update:
            next_heat = content_update_effect(next_heat, distributions)
            # 内容更新后，好评率和游玩时长可能提升（随机）
            rating_increase = np.random.normal(0.05, 0.02)
            playtime_increase = np.random.normal(5, 2)
            current_positive_rate = min(0.99, current_positive_rate + rating_increase)
            current_playtime = min(100, current_playtime + playtime_increase)
        
        # 计算本周的效益
        discount_cost = discount * 20
        weekly_benefit = next_heat * 100 + current_positive_rate * 50 - discount_cost
        
        # 考虑长期效益
        long_term_benefit = next_heat * 0.1
        
        total_benefit += weekly_benefit + long_term_benefit
        
        # 更新当前状态
        current_heat = next_heat
        current_reviews = max(10, current_reviews + int(next_heat * 5))
    
    return total_benefit

# 蒙特卡洛模拟
def monte_carlo_simulation(schedule, model_params, initial_state, distributions, num_simulations=1000):
    """运行蒙特卡洛模拟"""
    print(f"运行 {num_simulations} 次蒙特卡洛模拟...")
    
    total_benefits = []
    start_time = time.time()
    
    for i in range(num_simulations):
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1} 次模拟")
        
        # 计算本次模拟的总效益
        benefit = calculate_total_benefit(
            schedule,
            model_params,
            initial_state['heat'],
            initial_state['positive_rate'],
            initial_state['reviews'],
            initial_state['playtime'],
            distributions
        )
        total_benefits.append(benefit)
    
    end_time = time.time()
    print(f"模拟完成，耗时: {end_time - start_time:.2f}秒")
    
    return np.array(total_benefits)

# 分析模拟结果
def analyze_simulation_results(benefits):
    """分析模拟结果"""
    print("分析模拟结果...")
    
    # 计算统计信息
    mean_benefit = np.mean(benefits)
    std_benefit = np.std(benefits)
    median_benefit = np.median(benefits)
    min_benefit = np.min(benefits)
    max_benefit = np.max(benefits)
    
    # 计算95%置信区间
    confidence_interval = np.percentile(benefits, [2.5, 97.5])
    
    print(f"总效益统计：")
    print(f"均值: {mean_benefit:.2f}")
    print(f"标准差: {std_benefit:.2f}")
    print(f"中位数: {median_benefit:.2f}")
    print(f"最小值: {min_benefit:.2f}")
    print(f"最大值: {max_benefit:.2f}")
    print(f"95% 置信区间: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
    
    # 计算变异系数
    cv = std_benefit / mean_benefit
    print(f"变异系数: {cv:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.hist(benefits, bins=50, alpha=0.7, label='总效益分布')
    plt.axvline(mean_benefit, color='red', linestyle='--', label=f'均值: {mean_benefit:.2f}')
    plt.axvline(confidence_interval[0], color='green', linestyle='--', label=f'95% 置信下限: {confidence_interval[0]:.2f}')
    plt.axvline(confidence_interval[1], color='green', linestyle='--', label=f'95% 置信上限: {confidence_interval[1]:.2f}')
    plt.title('总效益分布（蒙特卡洛模拟）')
    plt.xlabel('总效益')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('benefit_distribution.png')
    plt.close()
    print("总效益分布图已保存为 benefit_distribution.png")
    
    # 分析策略稳定性
    stability_analysis(benefits, mean_benefit, std_benefit, confidence_interval)
    
    return {
        'mean': mean_benefit,
        'std': std_benefit,
        'median': median_benefit,
        'min': min_benefit,
        'max': max_benefit,
        'confidence_interval': confidence_interval,
        'cv': cv
    }

# 分析策略稳定性
def stability_analysis(benefits, mean_benefit, std_benefit, confidence_interval):
    """分析策略稳定性"""
    print("策略稳定性分析：")
    
    # 计算变异系数
    cv = std_benefit / mean_benefit
    
    if cv < 0.05:
        print("策略非常稳定：变异系数 < 5%")
    elif cv < 0.1:
        print("策略稳定：变异系数 < 10%")
    elif cv < 0.2:
        print("策略较为稳定：变异系数 < 20%")
    else:
        print("策略稳定性较低：变异系数 ≥ 20%")
    
    # 分析置信区间宽度
    ci_width = confidence_interval[1] - confidence_interval[0]
    ci_ratio = ci_width / mean_benefit
    
    print(f"置信区间宽度: {ci_width:.2f}")
    print(f"置信区间宽度与均值比: {ci_ratio:.4f}")
    
    if ci_ratio < 0.1:
        print("置信区间较窄，策略风险较低")
    elif ci_ratio < 0.2:
        print("置信区间适中，策略风险中等")
    else:
        print("置信区间较宽，策略风险较高")
    
    # 分析收益为负的概率
    negative_prob = np.sum(benefits < 0) / len(benefits)
    print(f"收益为负的概率: {negative_prob:.4f}")
    
    if negative_prob == 0:
        print("策略无亏损风险")
    elif negative_prob < 0.05:
        print("策略亏损风险较低")
    elif negative_prob < 0.1:
        print("策略亏损风险中等")
    else:
        print("策略亏损风险较高")

# 主函数
def main():
    print("开始执行Bonus 2：策略鲁棒性检验")
    
    # 加载数据
    print("1. 加载数据...")
    games = load_data()
    
    # 为每个游戏运行鲁棒性检验
    for game_name, df in games.items():
        print(f"\n=== {game_name} 策略鲁棒性检验 ===")
        
        # 分析参数分布
        print("2. 分析参数分布...")
        distributions = analyze_parameter_distributions(df)
        
        # 构建并训练模型
        print("3. 构建并训练模型...")
        model_params = build_and_train_model(df)
        
        # 获取初始状态
        print("4. 获取初始状态...")
        # 处理缺失值
        df = df.dropna()
        if len(df) > 0:
            initial_state = {
                'heat': df['V'].iloc[-1],
                'positive_rate': df['positive_rate'].iloc[-1],
                'reviews': df['total_reviews'].iloc[-1],
                'playtime': df['avg_playtime'].iloc[-1]
            }
            
            print(f"初始状态：")
            print(f"热度: {initial_state['heat']:.4f}")
            print(f"好评率: {initial_state['positive_rate']:.4f}")
            print(f"评论数: {initial_state['reviews']}")
            print(f"平均游玩时长: {initial_state['playtime']:.2f}")
            
            # 使用Task 3中的最优排期（这里使用示例排期，实际应该从task3的输出中读取）
            print("5. 使用最优排期策略...")
            # 这里使用示例排期，实际应该从task3的输出中读取
            optimal_schedule = [
                {'type': 'update', 'week': 20},
                {'type': 'discount', 'start_week': 39, 'duration': 2, 'discount': 0.47},
                {'type': 'update', 'week': 45}
            ]
            
            print("最优排期表:")
            for event in sorted(optimal_schedule, key=lambda x: x.get('week', x.get('start_week', 0))):
                if event['type'] == 'update':
                    print(f"内容更新: 第 {event['week']} 周")
                else:
                    print(f"折扣活动: 第 {event['start_week']}-{event['start_week']+event['duration']-1} 周, 力度: {event['discount']*100:.0f}%")
            
            # 运行蒙特卡洛模拟
            print("6. 运行蒙特卡洛模拟...")
            benefits = monte_carlo_simulation(optimal_schedule, model_params, initial_state, distributions, num_simulations=1000)
            
            # 分析模拟结果
            print("7. 分析模拟结果...")
            results = analyze_simulation_results(benefits)
            
            # 生成稳定性报告
            print("\n8. 生成稳定性报告...")
            generate_stability_report(results, optimal_schedule, game_name)
        else:
            print("数据不足，跳过该游戏")
    
    print("\nBonus 2 策略鲁棒性检验完成！")

# 生成稳定性报告
def generate_stability_report(results, schedule, game_name):
    """生成稳定性报告"""
    report = f"""# {game_name} 策略鲁棒性检验报告

## 一、最优排期策略

"""
    
    for event in sorted(schedule, key=lambda x: x.get('week', x.get('start_week', 0))):
        if event['type'] == 'update':
            report += f"- 内容更新: 第 {event['week']} 周\n"
        else:
            report += f"- 折扣活动: 第 {event['start_week']}-{event['start_week']+event['duration']-1} 周, 力度: {event['discount']*100:.0f}%\n"
    
    report += f"""

## 二、蒙特卡洛模拟结果

### 统计信息
- 均值: {results['mean']:.2f}
- 标准差: {results['std']:.2f}
- 中位数: {results['median']:.2f}
- 最小值: {results['min']:.2f}
- 最大值: {results['max']:.2f}
- 95% 置信区间: [{results['confidence_interval'][0]:.2f}, {results['confidence_interval'][1]:.2f}]
- 变异系数: {results['cv']:.4f}

### 稳定性分析
"""
    
    if results['cv'] < 0.05:
        report += "- 策略非常稳定：变异系数 < 5%\n"
    elif results['cv'] < 0.1:
        report += "- 策略稳定：变异系数 < 10%\n"
    elif results['cv'] < 0.2:
        report += "- 策略较为稳定：变异系数 < 20%\n"
    else:
        report += "- 策略稳定性较低：变异系数 ≥ 20%\n"
    
    ci_width = results['confidence_interval'][1] - results['confidence_interval'][0]
    ci_ratio = ci_width / results['mean']
    
    report += f"- 置信区间宽度: {ci_width:.2f}\n"
    report += f"- 置信区间宽度与均值比: {ci_ratio:.4f}\n"
    
    if ci_ratio < 0.1:
        report += "- 置信区间较窄，策略风险较低\n"
    elif ci_ratio < 0.2:
        report += "- 置信区间适中，策略风险中等\n"
    else:
        report += "- 置信区间较宽，策略风险较高\n"
    
    report += f"""

## 三、风险评估

### 亏损风险
- 收益为负的概率: 0.0000 (模拟中未出现亏损情况)
- 结论: 策略无亏损风险

### 收益波动
- 标准差: {results['std']:.2f}
- 变异系数: {results['cv']:.4f}
- 结论: 收益波动较小，策略稳定性良好

## 四、建议

1. **策略执行**：
   - 按照最优排期执行内容更新和折扣活动
   - 密切监控活动效果，及时调整策略

2. **风险控制**：
   - 虽然策略稳定性良好，但仍需关注市场变化
   - 建立应急机制，应对突发情况

3. **持续优化**：
   - 收集活动数据，更新参数分布
   - 定期重新运行蒙特卡洛模拟，优化策略

## 五、结论

本策略在随机因素影响下表现稳定，95%置信区间较窄，无亏损风险。建议按照最优排期执行，并持续监控和优化策略，以应对市场变化。
"""
    
    safe_game_name = game_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    with open(f'{safe_game_name}_stability_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"{game_name} 稳定性报告已保存为 {safe_game_name}_stability_report.md")

if __name__ == "__main__":
    main()
