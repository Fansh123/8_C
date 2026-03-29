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

# 预测热度
def predict_heat(model_params, V_t, positive_rate, total_reviews, avg_playtime, discount=0):
    """预测下一周的热度"""
    # 考虑折扣对热度的影响（假设折扣力度越大，热度提升越多）
    # 折扣力度范围：0-0.9（0表示无折扣，0.9表示90%折扣）
    discount_effect = discount * 0.3  # 折扣对热度的最大影响为0.3
    
    # 构建特征向量
    X = np.array([[V_t, positive_rate, total_reviews, avg_playtime]])
    
    # 基础预测（使用非线性模型）
    alpha, eta, delta, beta0, gamma1, gamma2, gamma3, intercept = model_params
    base_pred = alpha * V_t + eta * (V_t ** delta) + beta0 * discount + gamma1 * positive_rate + gamma2 * total_reviews + gamma3 * avg_playtime + intercept
    
    # 应用折扣效果
    final_pred = base_pred + discount_effect
    
    # 确保热度在合理范围内
    final_pred = max(0, min(1, final_pred))
    
    return final_pred

# 内容更新对热度的影响
def content_update_effect(base_heat):
    """内容更新对热度的影响"""
    # 内容更新可以提升热度
    update_effect = 0.2  # 内容更新对热度的影响
    
    # 应用更新效果
    new_heat = base_heat + update_effect
    
    # 确保热度在合理范围内
    new_heat = max(0, min(1, new_heat))
    
    return new_heat

# 定义目标函数
def objective_function(schedule, model_params, initial_heat, initial_positive_rate, initial_reviews, initial_playtime):
    """目标函数：最大化游戏厂家的效益"""
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
        
        # 预测下一周的热度
        next_heat = predict_heat(model_params, current_heat, current_positive_rate, current_reviews, current_playtime, discount)
        
        # 应用内容更新效果
        if is_update:
            next_heat = content_update_effect(next_heat)
            # 内容更新后，好评率和游玩时长可能提升
            current_positive_rate = min(0.99, current_positive_rate + 0.05)
            current_playtime = min(100, current_playtime + 5)
        
        # 计算本周的效益
        # 效益 = 热度 * 100 + 好评率 * 50 - 折扣成本
        discount_cost = discount * 20  # 折扣会减少收益
        weekly_benefit = next_heat * 100 + current_positive_rate * 50 - discount_cost
        
        # 考虑长期效益（热度持续性）
        long_term_benefit = next_heat * 0.1
        
        total_benefit += weekly_benefit + long_term_benefit
        
        # 更新当前状态
        current_heat = next_heat
        # 假设评论数随热度增加而增加
        current_reviews = max(10, current_reviews + int(next_heat * 5))
    
    return total_benefit

# 检查约束条件
def check_constraints(schedule):
    """检查排期是否满足约束条件"""
    # 1. 折扣约束
    discount_events = [event for event in schedule if event['type'] == 'discount']
    for event in discount_events:
        if event['discount'] > 0.9:
            return False, "折扣力度超过90%"
        if event['duration'] > 2:
            return False, "折扣持续时长超过2周"
    
    # 2. 折扣间隔约束
    discount_weeks = sorted([event['start_week'] for event in discount_events])
    for i in range(len(discount_weeks) - 1):
        if discount_weeks[i+1] - (discount_weeks[i] + 2) < 4:
            return False, "折扣活动间隔不足4周"
    
    # 3. 内容更新约束
    update_events = [event for event in schedule if event['type'] == 'update']
    if len(update_events) > 2:
        return False, "每年内容更新次数超过2次"
    
    update_weeks = sorted([event['week'] for event in update_events])
    for i in range(len(update_weeks) - 1):
        if update_weeks[i+1] - update_weeks[i] < 12:  # 12周 = 3个月
            return False, "内容更新间隔不足3个月"
    
    return True, "满足所有约束条件"

# 生成初始排期
def generate_initial_schedule():
    """生成初始排期表"""
    schedule = []
    
    # 生成最多2次内容更新
    update_count = random.randint(0, 2)
    update_weeks = []
    for i in range(update_count):
        week = random.randint(0, 51)
        # 确保更新间隔至少12周
        if i > 0 and week - update_weeks[-1] < 12:
            week = update_weeks[-1] + 12
        update_weeks.append(week)
        schedule.append({'type': 'update', 'week': week})
    
    # 生成最多3次折扣活动
    discount_count = random.randint(0, 3)
    discount_weeks = []
    for i in range(discount_count):
        start_week = random.randint(0, 49)
        duration = random.randint(1, 2)
        discount = random.uniform(0.1, 0.9)
        
        # 确保折扣间隔至少4周
        valid = True
        for existing in discount_weeks:
            if abs(start_week - existing) < 4:
                valid = False
                break
        
        if valid:
            discount_weeks.append(start_week)
            schedule.append({
                'type': 'discount',
                'start_week': start_week,
                'duration': duration,
                'discount': discount
            })
    
    return schedule

# 遗传算法优化
def genetic_algorithm(model_params, initial_heat, initial_positive_rate, initial_reviews, initial_playtime, 
                     population_size=50, generations=100, mutation_rate=0.1):
    """使用遗传算法求解最优排期"""
    # 生成初始种群
    population = []
    for _ in range(population_size):
        schedule = generate_initial_schedule()
        valid, _ = check_constraints(schedule)
        if valid:
            fitness = objective_function(schedule, model_params, initial_heat, initial_positive_rate, initial_reviews, initial_playtime)
            population.append((schedule, fitness))
    
    # 确保种群大小
    while len(population) < population_size:
        schedule = generate_initial_schedule()
        valid, _ = check_constraints(schedule)
        if valid:
            fitness = objective_function(schedule, model_params, initial_heat, initial_positive_rate, initial_reviews, initial_playtime)
            population.append((schedule, fitness))
    
    # 进化过程
    for generation in range(generations):
        # 按适应度排序
        population.sort(key=lambda x: x[1], reverse=True)
        
        # 保留精英
        elite_size = int(population_size * 0.2)
        new_population = population[:elite_size]
        
        # 交叉和变异
        while len(new_population) < population_size:
            # 选择父母
            parent1 = random.choice(population[:int(population_size * 0.5)])[0]
            parent2 = random.choice(population[:int(population_size * 0.5)])[0]
            
            # 交叉
            child = []
            # 合并并去重
            all_events = parent1 + parent2
            event_set = set()
            for event in all_events:
                if event['type'] == 'update':
                    key = (event['type'], event['week'])
                else:
                    key = (event['type'], event['start_week'])
                if key not in event_set:
                    event_set.add(key)
                    child.append(event)
            
            # 变异
            if random.random() < mutation_rate:
                # 随机添加或修改事件
                if random.random() < 0.5:
                    # 添加事件
                    if random.random() < 0.5 and len([e for e in child if e['type'] == 'update']) < 2:
                        # 添加内容更新
                        week = random.randint(0, 51)
                        child.append({'type': 'update', 'week': week})
                    else:
                        # 添加折扣活动
                        start_week = random.randint(0, 49)
                        duration = random.randint(1, 2)
                        discount = random.uniform(0.1, 0.9)
                        child.append({
                            'type': 'discount',
                            'start_week': start_week,
                            'duration': duration,
                            'discount': discount
                        })
                else:
                    # 修改事件
                    if child:
                        event_idx = random.randint(0, len(child) - 1)
                        event = child[event_idx]
                        if event['type'] == 'update':
                            event['week'] = random.randint(0, 51)
                        else:
                            event['start_week'] = random.randint(0, 49)
                            event['duration'] = random.randint(1, 2)
                            event['discount'] = random.uniform(0.1, 0.9)
            
            # 检查约束
            valid, _ = check_constraints(child)
            if valid:
                fitness = objective_function(child, model_params, initial_heat, initial_positive_rate, initial_reviews, initial_playtime)
                new_population.append((child, fitness))
        
        population = new_population
        
        # 打印进度
        if (generation + 1) % 10 == 0:
            best_fitness = population[0][1]
            print(f"第 {generation + 1} 代，最佳适应度: {best_fitness:.2f}")
    
    # 返回最佳排期
    population.sort(key=lambda x: x[1], reverse=True)
    return population[0]

# 可视化排期结果
def plot_schedule(schedule, game_name):
    """可视化排期结果"""
    plt.figure(figsize=(15, 6))
    
    # 绘制时间轴
    weeks = list(range(52))
    plt.plot(weeks, [0]*52, 'k-', alpha=0.3)
    
    # 绘制折扣活动
    for event in schedule:
        if event['type'] == 'discount':
            start = event['start_week']
            end = start + event['duration']
            discount = event['discount']
            plt.fill_between(range(start, end), 0, discount*100, alpha=0.5, label=f"折扣: {discount*100:.0f}%")
    
    # 绘制内容更新
    for event in schedule:
        if event['type'] == 'update':
            week = event['week']
            plt.plot(week, 50, 'ro', markersize=10, label="内容更新")
    
    plt.title(f'{game_name} 12个月最优排期表')
    plt.xlabel('周数')
    plt.ylabel('折扣力度 (%)')
    plt.ylim(-10, 100)
    plt.xticks(range(0, 52, 4))
    plt.grid(alpha=0.3)
    
    # 去重图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.savefig(f'{game_name}_optimal_schedule.png')
    plt.close()
    print(f"{game_name} 最优排期表已保存为 {game_name}_optimal_schedule.png")

# 主函数
def main():
    print("开始执行Task 3：运筹优化")
    
    # 加载数据
    print("1. 加载数据...")
    games = load_data()
    print(f"数据加载完成，共 {len(games)} 个游戏")
    
    # 处理所有可用的游戏
    for game_name, df in games.items():
        print(f"\n=== {game_name} 运筹优化 ===")
        
        # 构建并训练模型
        print("2. 构建并训练模型...")
        model_params = build_and_train_model(df)
        print("模型训练完成")
        
        # 获取初始状态
        print("3. 获取初始状态...")
        # 处理缺失值
        df = df.dropna()
        if len(df) > 0:
            initial_heat = df['V'].iloc[-1]
            initial_positive_rate = df['positive_rate'].iloc[-1]
            initial_reviews = df['total_reviews'].iloc[-1]
            initial_playtime = df['avg_playtime'].iloc[-1]
            
            print(f"初始状态：")
            print(f"热度: {initial_heat:.4f}")
            print(f"好评率: {initial_positive_rate:.4f}")
            print(f"评论数: {initial_reviews}")
            print(f"平均游玩时长: {initial_playtime:.2f}")
            
            # 使用遗传算法求解最优排期
            print("\n4. 使用遗传算法求解最优排期...")
            start_time = time.time()
            # 减少种群大小和代数，加快计算速度
            best_schedule, best_fitness = genetic_algorithm(
                model_params, initial_heat, initial_positive_rate, initial_reviews, initial_playtime,
                population_size=20, generations=50
            )
            end_time = time.time()
            
            print(f"\n优化完成，耗时: {end_time - start_time:.2f}秒")
            print(f"最佳适应度: {best_fitness:.2f}")
            
            # 打印最优排期
            print("\n5. 最优排期表:")
            for event in sorted(best_schedule, key=lambda x: x.get('week', x.get('start_week', 0))):
                if event['type'] == 'update':
                    print(f"内容更新: 第 {event['week']} 周")
                else:
                    print(f"折扣活动: 第 {event['start_week']}-{event['start_week']+event['duration']-1} 周, 力度: {event['discount']*100:.0f}%")
            
            # 可视化排期结果
            print("\n6. 可视化排期结果...")
            plot_schedule(best_schedule, game_name)
            print("可视化完成")
        else:
            print("数据不足，跳过该游戏")
    
    print("\nTask 3 运筹优化完成！")

if __name__ == "__main__":
    main()
