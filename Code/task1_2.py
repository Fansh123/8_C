import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# 设置matplotlib为非交互模式
plt.switch_backend('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 从pachong.py生成的reviews文件加载数据
def load_data_from_pachong():
    """从pachong.py生成的CSV文件加载游戏数据"""
    games = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找所有评论数据文件
    review_files = []
    for file in os.listdir(current_dir):
        if file.startswith('reviews_') and file.endswith('.csv'):
            review_files.append(file)
    
    if not review_files:
        print("警告：未找到pachong.py生成的reviews文件，请确保已运行pachong.py")
        return {}
    
    print(f"找到 {len(review_files)} 个游戏评论数据文件")
    
    for file in review_files:
        try:
            # 从文件名提取游戏ID
            app_id = file.split('_')[1]
            
            # 读取CSV文件
            df = pd.read_csv(os.path.join(current_dir, file), encoding='utf-8-sig')
            
            # 确保时间戳被正确解析
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('date', inplace=True)
            
            # 将游玩时长从分钟转换为小时
            df['playtime_at_review'] = df['playtime_at_review'] / 60
            
            # 按周聚合数据
            weekly = df.resample('W').agg({
                'comment': 'count',                     # 评论数
                'voted_up': lambda x: x.mean() if len(x) > 0 else np.nan,  # 好评率
                'playtime_at_review': 'mean'            # 平均游玩时长（小时）
            }).rename(columns={
                'comment': 'total_reviews',
                'voted_up': 'positive_rate',
                'playtime_at_review': 'avg_playtime'
            })
            
            # 处理无评论的周：评论数填0，好评率和平均时长填NaN
            weekly['total_reviews'] = weekly['total_reviews'].fillna(0)
            
            # 获取游戏名称
            game_name = df['game_name'].iloc[0] if 'game_name' in df.columns else f"Game_{app_id}"
            
            games[game_name] = weekly
            print(f"  成功加载 {game_name} ({app_id}): {len(weekly)} 条周数据")
            
        except Exception as e:
            print(f"  加载文件 {file} 失败: {e}")
    
    return games

# 熵权法计算权重
def entropy_weight(df):
    """计算熵权法权重，df 是已归一化的数据框，每列是指标"""
    # 计算第 j 个指标下第 i 个样本的权重
    p = df.div(df.sum(axis=0), axis=1)
    # 计算熵值
    e = - (p * np.log(p + 1e-12)).sum(axis=0) / np.log(len(df))
    # 计算差异系数
    d = 1 - e
    # 计算权重
    w = d / d.sum()
    return w

# 计算复合热度指标
def calculate_heat_index(games):
    """计算每个游戏的复合热度指标"""
    results = {}
    
    for game_name, game_df in games.items():
        # 选取三个指标
        features = game_df[['total_reviews', 'positive_rate', 'avg_playtime']]
        
        # 数据标准化
        scaler = MinMaxScaler()
        norm_features = scaler.fit_transform(features)
        df_norm = pd.DataFrame(norm_features, columns=features.columns, index=features.index)
        
        # 计算权重
        weights = entropy_weight(df_norm)
        print(f"\n{game_name} 指标权重:")
        print(weights)
        
        # 计算复合热度
        game_df['V'] = (df_norm * weights).sum(axis=1)
        results[game_name] = game_df
    
    return results

# 可视化热度曲线
def plot_heat_curves(results):
    """绘制多款游戏的热度曲线"""
    plt.figure(figsize=(14, 7))
    
    for game_name, game_df in results.items():
        plt.plot(game_df.index, game_df['V'], marker='o', linestyle='-', linewidth=1, markersize=3, label=game_name)
    
    plt.title('Game Popularity Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Composite Popularity Index', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('game_popularity.png')
    plt.show()
    
    print("\n热度曲线已保存为 game_popularity.png")

# 主函数
def main():
    print("从pachong.py生成的数据文件加载数据...")
    games = load_data_from_pachong()
    
    if not games:
        print("\n错误：未能加载任何游戏数据，请确保：")
        print("1. 已运行pachong.py生成数据文件")
        print("2. 数据文件(*_weekly_heat.csv)与task1_2.py在同一目录")
        return
    
    print(f"\n成功加载 {len(games)} 个游戏的数据")
    
    print("\n计算复合热度指标...")
    results = calculate_heat_index(games)
    
    print("\n可视化热度曲线...")
    plot_heat_curves(results)
    
    # 保存结果
    for game_name, game_df in results.items():
        # 将空格替换为下划线以生成有效的文件名
        safe_name = game_name.replace(' ', '_')
        game_df.to_csv(f'{safe_name}_weekly_data.csv')
        print(f"\n{game_name} 的周数据已保存为 {safe_name}_weekly_data.csv")

if __name__ == "__main__":
    main()