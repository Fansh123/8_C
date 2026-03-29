import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import os

# 设置matplotlib为非交互模式，确保在无图形界面环境中也能生成图表
plt.switch_backend('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
            # 读取CSV文件
            df = pd.read_csv(os.path.join(current_dir, file), encoding='utf-8-sig')
            
            # 确保时间戳被正确解析
            df['date'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # 重命名列以匹配task1_3的格式
            if 'playtime_at_review' in df.columns:
                df = df.rename(columns={'playtime_at_review': 'playtime'})
            
            # 获取游戏名称
            game_name = df['game_name'].iloc[0] if 'game_name' in df.columns else f"Game_{file.split('_')[1]}"
            
            # 只保留需要的列
            df = df[['date', 'playtime', 'voted_up']]
            
            # 将游玩时长从分钟转换为小时
            df['playtime'] = df['playtime'] / 60
            
            games[game_name] = df
            print(f"  成功加载 {game_name}: {len(df)} 条评论数据")
            
        except Exception as e:
            print(f"  加载文件 {file} 失败: {e}")
    
    return games

# 分析游玩时长分布
def analyze_playtime_distribution(games):
    """分析游玩时长分布并尝试拟合理论分布"""
    for game_name, df in games.items():
        print(f"\n=== {game_name} 游玩时长分布分析 ===")
        
        # 基本统计信息
        playtime = df['playtime']
        print(f"平均值: {playtime.mean():.2f} 小时")
        print(f"中位数: {playtime.median():.2f} 小时")
        print(f"标准差: {playtime.std():.2f} 小时")
        print(f"最小值: {playtime.min():.2f} 小时")
        print(f"最大值: {playtime.max():.2f} 小时")
        
        # 绘制分布图
        plt.figure(figsize=(12, 6))
        
        # 直方图
        plt.subplot(1, 2, 1)
        sns.histplot(playtime, bins=50, kde=True)
        plt.title(f'{game_name} 游玩时长分布')
        plt.xlabel('游玩时长（小时）')
        plt.ylabel('频数')
        
        # Q-Q图，用于检验分布 - 先拟合分布再绘制
        plt.subplot(1, 2, 2)
        # 先拟合伽马分布
        shape, loc, scale = stats.gamma.fit(playtime)
        # 使用拟合的参数绘制Q-Q图
        stats.probplot(playtime, dist="gamma", sparams=(shape, loc, scale), plot=plt)
        plt.title(f'{game_name} Q-Q图（伽马分布）')
        
        plt.tight_layout()
        plt.savefig(f'{game_name}_playtime_distribution.png')
        plt.close()
        print(f"\n{game_name} 游玩时长分布图已保存为 {game_name}_playtime_distribution.png")
        
        # 拟合伽马分布参数（已在Q-Q图绘制时拟合）
        print(f"\n拟合伽马分布参数:")
        print(f"形状参数 (shape): {shape:.4f}")
        print(f"位置参数 (loc): {loc:.4f}")
        print(f"尺度参数 (scale): {scale:.4f}")
        
        # 计算拟合优度
        # 生成拟合分布的随机样本
        fitted_data = stats.gamma.rvs(shape, loc=loc, scale=scale, size=len(playtime))
        # 计算KS检验
        ks_stat, ks_pvalue = stats.kstest(playtime, 'gamma', args=(shape, loc, scale))
        print(f"\nKS检验结果:")
        print(f"统计量: {ks_stat:.4f}")
        print(f"p值: {ks_pvalue:.4f}")
        if ks_pvalue > 0.05:
            print("接受原假设：游玩时长服从伽马分布")
        else:
            print("拒绝原假设：游玩时长不服从伽马分布")
        
        # 保存分布数据到CSV
        playtime_dist = pd.DataFrame({'playtime': playtime})
        playtime_dist.to_csv(f'{game_name}_playtime_distribution.csv', index=False)
        print(f"\n{game_name} 游玩时长分布数据已保存为 {game_name}_playtime_distribution.csv")

# 分析好评率与游玩时长的关系
def analyze_playtime_vs_rating(games):
    """分析游玩时长与好评率的关系"""
    for game_name, df in games.items():
        print(f"\n=== {game_name} 游玩时长与好评率关系 ===")
        
        # 按游玩时长分组，计算每组的好评率
        df['playtime_group'] = pd.cut(df['playtime'], bins=[0, 10, 50, 100, 200, 500, np.inf], 
                                     labels=['<10h', '10-50h', '50-100h', '100-200h', '200-500h', '>500h'])
        
        rating_by_playtime = df.groupby('playtime_group')['voted_up'].mean()
        print(rating_by_playtime)
        
        # 可视化
        plt.figure(figsize=(10, 5))
        rating_by_playtime.plot(kind='bar')
        plt.title(f'{game_name} 不同游玩时长的好评率')
        plt.xlabel('游玩时长分组')
        plt.ylabel('好评率')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{game_name}_playtime_vs_rating.png')
        plt.close()
        print(f"\n{game_name} 游玩时长与好评率关系图已保存为 {game_name}_playtime_vs_rating.png")
        
        # 保存数据到CSV
        rating_by_playtime.to_csv(f'{game_name}_playtime_vs_rating.csv')
        print(f"\n{game_name} 游玩时长与好评率关系数据已保存为 {game_name}_playtime_vs_rating.csv")

# 分析评论活跃度随时间的变化
def analyze_review_activity(games):
    """分析评论活跃度随时间的变化"""
    print("\n=== 评论活跃度随时间变化分析 ===")
    
    # 保存所有游戏的周评论数据
    all_weekly_reviews = pd.DataFrame()
    
    for game_name, df in games.items():
        # 按周统计评论数
        df.set_index('date', inplace=True)
        weekly_reviews = df.resample('W').size()
        
        # 将数据添加到总数据框
        all_weekly_reviews[game_name] = weekly_reviews
        
        # 保存单个游戏的周评论数据
        weekly_reviews.to_csv(f'{game_name}_weekly_reviews.csv')
        print(f"{game_name} 周评论数据已保存为 {game_name}_weekly_reviews.csv")
    
    # 可视化所有游戏的评论活跃度
    plt.figure(figsize=(14, 8))
    
    for game_name in all_weekly_reviews.columns:
        plt.plot(all_weekly_reviews.index, all_weekly_reviews[game_name], label=game_name, linewidth=1.5)
    
    plt.title('游戏评论活跃度随时间变化')
    plt.xlabel('日期')
    plt.ylabel('每周评论数')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('review_activity_over_time.png')
    plt.close()
    print("\n游戏评论活跃度随时间变化图已保存为 review_activity_over_time.png")
    
    # 保存所有游戏的周评论数据
    all_weekly_reviews.to_csv('all_games_weekly_reviews.csv')
    print("所有游戏的周评论数据已保存为 all_games_weekly_reviews.csv")

# 主函数
def main():
    print("从pachong.py生成的数据文件加载数据...")
    games = load_data_from_pachong()
    
    if not games:
        print("\n错误：未能加载任何游戏数据，请确保：")
        print("1. 已运行pachong.py生成数据文件")
        print("2. 数据文件(reviews_*.csv)与task1_3.py在同一目录")
        return
    
    print(f"\n成功加载 {len(games)} 个游戏的数据")
    
    print("分析游玩时长分布...")
    analyze_playtime_distribution(games)
    
    print("分析游玩时长与好评率的关系...")
    analyze_playtime_vs_rating(games)
    
    print("分析评论活跃度随时间的变化...")
    analyze_review_activity(games)
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()