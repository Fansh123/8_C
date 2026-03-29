# -*- coding: utf-8 -*-
"""
Steam 数据采集与热度分析（完整版）
- 支持自动重试、会话管理、随机延迟，避免网络超时与限流
- 按周聚合，熵权法构建复合热度指标
- 游玩时长分布拟合（伽马分布）与可视化
- NLP 评论关键词分析（需安装 jieba）
"""

import os
import time
import random
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
# from steam_web_api import Steam  # 注释掉，使用requests直接调用API
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 设置 matplotlib 后端（无图形界面时可保存图片）
plt.switch_backend('Agg')
# 设置中文字体（Windows 用 SimHei，macOS/Linux 可改为 'WenQuanYi Micro Hei'）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
API_KEY = "7C21FA6FE3C60E8FE87A9890319FE0E8"          # 请替换为你的实际API密钥
GAME_IDS = [730,570,440]
REVIEWS_PER_GAME = 100000         # 目标评论数（至少10000）
REQUEST_DELAY = 1.5              # 基础请求间隔（秒）

# ==================== 辅助函数 ====================
def fetch_game_details(app_id):
    """
    通过 Steam Store API 获取游戏详细信息（带重试和超时）
    """
    url = "https://store.steampowered.com/api/appdetails"
    params = {
        "appids": app_id,
        "cc": "cn",                     # 可改为 "us" 获取美元价格
        "filters": "basic,price_overview,platforms,genres,metacritic,recommendations"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # 创建带重试的会话
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,               # 重试间隔指数增长：2, 4, 8 秒
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    try:
        response = session.get(url, params=params, headers=headers, timeout=(10, 30))
        if response.status_code != 200:
            print(f"  游戏 {app_id} 详情请求失败，状态码 {response.status_code}")
            return None
        data = response.json()
        if data and str(app_id) in data and data[str(app_id)]['success']:
            details = data[str(app_id)]['data']
            return {
                'app_id': app_id,
                'name': details.get('name'),
                'developers': ', '.join(details.get('developers', [])),
                'publishers': ', '.join(details.get('publishers', [])),
                'release_date': details.get('release_date', {}).get('date', 'N/A'),
                'price': details.get('price_overview', {}).get('final_formatted', '免费'),
                'platforms': ', '.join([p for p, v in details.get('platforms', {}).items() if v]),
                'genres': ', '.join([g['description'] for g in details.get('genres', [])]),
                'metacritic_score': details.get('metacritic', {}).get('score', 'N/A'),
                'recommendations': details.get('recommendations', {}).get('total', 'N/A')
            }
        else:
            return None
    except Exception as e:
        print(f"  获取游戏 {app_id} 详细信息时发生异常: {e}")
        return None
    finally:
        session.close()

def is_valid_review(text):
    """简单清洗：过滤过短、纯符号、重复字符等无意义评论"""
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 10:
        return False
    if all(c in "!@#$%^&*()_+=-[]{};:'\",.<>/?\\|~` " for c in text):
        return False
    if len(set(text)) == 1:
        return False
    # 可扩展过滤常见垃圾文案，如“疯狂星期四”等
    return True

def fetch_reviews(app_id, target_count=10000, max_pages=None):
    """
    获取 Steam 评论数据（带重试和会话管理）
    返回 DataFrame 包含：comment, timestamp, playtime_at_review, voted_up, votes_up, votes_funny
    """
    base_url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "filter": "all",           # 获取全部评论，确保时间跨度代表性
        "language": "all",
        "num_per_page": 100,
        "p": 1,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # 创建会话并配置重试策略
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    reviews = []
    cursor = "*"
    page = 1
    while True:
        params["cursor"] = cursor
        params["p"] = page
        try:
            response = session.get(base_url, params=params, headers=headers, timeout=(10, 30))
            if response.status_code != 200:
                print(f"  请求失败，状态码 {response.status_code}")
                if response.status_code == 429:
                    sleep_time = random.uniform(10, 20)
                    print(f"  触发限流，休眠 {sleep_time:.1f} 秒")
                    time.sleep(sleep_time)
                    continue
                break
            data = response.json()
        except requests.exceptions.Timeout as e:
            print(f"  请求超时，稍后重试... ({e})")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"  请求异常: {e}")
            break

        if not data.get("success"):
            print("  返回数据 success=False，停止采集")
            break

        review_list = data.get("reviews", [])
        if not review_list:
            break

        for rev in review_list:
            text = rev.get("review", "")
            if not is_valid_review(text):
                continue
            reviews.append({
                "comment": text,
                "timestamp": rev.get("timestamp_created"),
                "playtime_at_review": rev.get("author", {}).get("playtime_at_review", 0) / 60,  # 转换为小时
                "voted_up": rev.get("voted_up", False),
                "votes_up": rev.get("votes_up", 0),
                "votes_funny": rev.get("votes_funny", 0),
            })
        print(f"  已获取 {len(reviews)} 条有效评论（本页 {len(review_list)} 条）")

        if len(reviews) >= target_count:
            reviews = reviews[:target_count]
            break

        cursor = data.get("cursor")
        if not cursor:
            break
        page += 1
        if max_pages and page > max_pages:
            break

        # 随机延迟 1~3 秒，降低请求频率
        time.sleep(random.uniform(1.0, 3.0))

    session.close()
    return pd.DataFrame(reviews)

def entropy_weight(df_norm):
    """熵权法计算权重（df_norm 已归一化，列是指标）"""
    p = df_norm.div(df_norm.sum(axis=0), axis=1)
    e = - (p * np.log(p + 1e-12)).sum(axis=0) / np.log(len(df_norm))
    d = 1 - e
    w = d / d.sum()
    return w

def aggregate_weekly(df_reviews):
    """将评论数据按周聚合，返回周评论数、平均好评率、平均游玩时长"""
    df = df_reviews.copy()
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('date', inplace=True)
    # 按周重采样
    weekly = df.resample('W').agg({
        'comment': 'count',                     # 评论数
        'voted_up': lambda x: x.mean() if len(x) > 0 else np.nan,  # 好评率
        'playtime_at_review': 'mean'            # 平均游玩时长（小时）
    }).rename(columns={
        'comment': 'review_count',
        'voted_up': 'positive_rate',
        'playtime_at_review': 'avg_playtime'
    })
    # 处理无评论的周：评论数填0，好评率和平均时长填NaN
    weekly['review_count'] = weekly['review_count'].fillna(0)
    return weekly

# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("Task 1.1：数据采集")
    print("=" * 60)

    # 存储所有游戏的原始评论和基本信息
    games_data = {}
    games_info = []

    # 1. 获取游戏基本信息
    for app_id in GAME_IDS:
        print(f"\n获取游戏 {app_id} 基本信息...")
        info = fetch_game_details(app_id)
        if info:
            games_info.append(info)
            games_data[app_id] = {'info': info}
        else:
            # 创建占位符，防止 KeyError
            games_data[app_id] = {'info': {'name': f"Game_{app_id}", 'app_id': app_id}}
            print(f"  警告：游戏 {app_id} 基本信息获取失败，将使用临时名称")
        time.sleep(0.5)

    # 保存成功获取的游戏信息
    if games_info:
        df_games = pd.DataFrame(games_info)
        df_games.to_csv('steam_games_info.csv', index=False, encoding='utf-8-sig')
        print("\n游戏信息已保存至 steam_games_info.csv")
    else:
        print("\n警告：未成功获取任何游戏信息，后续分析可能受限")

    # 2. 采集评论数据
    for app_id in GAME_IDS:
        # 安全获取游戏名
        game_name = games_data[app_id]['info'].get('name', f"Game_{app_id}")
        print(f"\n正在采集游戏 {app_id} ({game_name}) 的评论...")
        df_reviews = fetch_reviews(app_id, target_count=REVIEWS_PER_GAME)
        if len(df_reviews) == 0:
            print(f"  警告：未获取到任何评论")
            continue
        # 添加游戏标识
        df_reviews['app_id'] = app_id
        df_reviews['game_name'] = game_name
        # 清洗文件名
        safe_name = game_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        filename = f"reviews_{app_id}_{safe_name}.csv"
        df_reviews.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"  已保存 {len(df_reviews)} 条有效评论至 {filename}")
        games_data[app_id]['reviews'] = df_reviews

    print("\n" + "=" * 60)
    print("Task 1.2：构建复合热度指标")
    print("=" * 60)

    # 3. 按周聚合并计算复合热度 V(t)
    heat_results = {}      # 存储每个游戏的周数据
    for app_id, data in games_data.items():
        if 'reviews' not in data:
            print(f"游戏 {app_id} 无评论数据，跳过热度计算")
            continue
        df_rev = data['reviews']
        weekly = aggregate_weekly(df_rev)
        if weekly.empty:
            print(f"游戏 {app_id} 无有效周数据，跳过")
            continue
        # 剔除没有评论的周（评论数为0）参与权重计算，因为这些周好评率和平均时长缺失
        valid_weeks = weekly[weekly['review_count'] > 0].copy()
        if len(valid_weeks) < 2:
            print(f"游戏 {app_id} 有效周数不足，跳过热度计算")
            continue
        # 选取指标（评论数、好评率、平均游玩时长）
        features = valid_weeks[['review_count', 'positive_rate', 'avg_playtime']].copy()
        # 缺失值处理（理论上 valid_weeks 已无缺失，但确保安全）
        features.fillna(method='ffill', inplace=True)
        features.fillna(0, inplace=True)
        # 标准化（Min-Max）
        scaler = MinMaxScaler()
        norm_features = scaler.fit_transform(features)
        df_norm = pd.DataFrame(norm_features, columns=features.columns, index=features.index)
        # 计算权重
        weights = entropy_weight(df_norm)
        print(f"\n游戏 {data['info'].get('name', f'Game_{app_id}')} (app_id={app_id}) 熵权法权重：")
        print(weights)
        # 计算复合热度 V(t)（对原 weekly 的所有行，用权重计算）
        # 先对全 weekly 进行同样的归一化（注意要用 valid_weeks 的 scaler 来转换）
        full_norm = scaler.transform(weekly[['review_count', 'positive_rate', 'avg_playtime']].fillna(0))
        full_df_norm = pd.DataFrame(full_norm, columns=features.columns, index=weekly.index)
        weekly['V'] = (full_df_norm * weights).sum(axis=1)
        # 对于原始评论数为0的周，V 值可能因归一化而失真，此处将其设为 NaN，后续插值
        weekly.loc[weekly['review_count'] == 0, 'V'] = np.nan
        # 线性插值填充 NaN
        weekly['V'] = weekly['V'].interpolate(method='time', limit_area='inside')
        heat_results[app_id] = weekly
        # 保存周数据
        safe_name = data['info'].get('name', f"Game_{app_id}").replace(' ', '_').replace('/', '_').replace('\\', '_')
        weekly.to_csv(f"{safe_name}_weekly_heat.csv", encoding='utf-8-sig')

    # 4. 绘制热度曲线
    if heat_results:
        plt.figure(figsize=(14, 7))
        for app_id, weekly in heat_results.items():
            game_name = games_data[app_id]['info'].get('name', f"Game_{app_id}")
            plt.plot(weekly.index, weekly['V'], marker='o', markersize=3, linewidth=1, label=game_name)
        plt.title('游戏复合热度指数随时间变化', fontsize=16)
        plt.xlabel('日期')
        plt.ylabel('复合热度指数 V(t)')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('game_heat_curves.png')
        plt.close()
        print("\n热度曲线已保存至 game_heat_curves.png")
    else:
        print("\n没有足够数据绘制热度曲线")

    print("\n" + "=" * 60)
    print("Task 1.3：数据洞察——游玩时长分布分析")
    print("=" * 60)

    # 5. 分析游玩时长分布
    for app_id, data in games_data.items():
        if 'reviews' not in data:
            continue
        df_rev = data['reviews']
        game_name = data['info'].get('name', f"Game_{app_id}")
        playtime = df_rev['playtime_at_review']
        if len(playtime) == 0:
            print(f"{game_name} 无游玩时长数据，跳过")
            continue

        print(f"\n=== {game_name} 游玩时长分析 ===")
        print(f"样本量：{len(playtime)}")
        print(f"平均值：{playtime.mean():.2f} 小时")
        print(f"中位数：{playtime.median():.2f} 小时")
        print(f"标准差：{playtime.std():.2f} 小时")
        print(f"最小值：{playtime.min():.2f} 小时")
        print(f"最大值：{playtime.max():.2f} 小时")

        # 绘制直方图 + 拟合伽马分布（剔除0时长）
        playtime_pos = playtime[playtime > 0]  # 伽马分布要求正值
        if len(playtime_pos) < 10:
            print("  有效正时长不足，跳过分布拟合")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # 直方图
        sns.histplot(playtime_pos, bins=50, kde=True, ax=axes[0])
        axes[0].set_title(f'{game_name} 游玩时长分布（剔除0）')
        axes[0].set_xlabel('游玩时长（小时）')
        axes[0].set_ylabel('频数')
        # Q-Q 图（拟合伽马分布）
        shape, loc, scale = stats.gamma.fit(playtime_pos)
        stats.probplot(playtime_pos, dist="gamma", sparams=(shape, loc, scale), plot=axes[1])
        axes[1].set_title(f'{game_name} Q-Q 图（伽马分布）')
        plt.tight_layout()
        safe_name = game_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        plt.savefig(f'{safe_name}_playtime_analysis.png')
        plt.close()

        # 拟合参数与 KS 检验
        print(f"\n拟合伽马分布参数：shape={shape:.4f}, loc={loc:.4f}, scale={scale:.4f}")
        ks_stat, ks_pvalue = stats.kstest(playtime_pos, 'gamma', args=(shape, loc, scale))
        print(f"KS 检验统计量：{ks_stat:.4f}, p值：{ks_pvalue:.4f}")
        if ks_pvalue > 0.05:
            print("接受原假设：游玩时长服从伽马分布")
        else:
            print("拒绝原假设：游玩时长不服从伽马分布")

        # 保存分布数据
        pd.DataFrame({'playtime': playtime}).to_csv(f'{safe_name}_playtime_dist.csv', index=False)

        # 按时间分析游玩时长变化趋势
        df_rev['date'] = pd.to_datetime(df_rev['timestamp'], unit='s')
        df_rev.set_index('date', inplace=True)
        weekly_playtime = df_rev['playtime_at_review'].resample('W').mean()
        plt.figure(figsize=(12, 5))
        plt.plot(weekly_playtime.index, weekly_playtime.values, marker='o', markersize=3)
        plt.title(f'{game_name} 平均游玩时长周变化')
        plt.xlabel('日期')
        plt.ylabel('平均游玩时长（小时）')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{safe_name}_weekly_playtime.png')
        plt.close()

    print("\n" + "=" * 60)
    print("Bonus 1：基于 NLP 的评论归因分析（示例）")
    print("=" * 60)

    # 6. 简单文本分析（需要 jieba）
    try:
        import jieba
        from collections import Counter

        # 选取一个游戏的评论进行简单分析
        sample_app_id = GAME_IDS[0]
        if sample_app_id in games_data and 'reviews' in games_data[sample_app_id]:
            df_rev = games_data[sample_app_id]['reviews']
            game_name = games_data[sample_app_id]['info'].get('name', f"Game_{sample_app_id}")
            print(f"\n分析游戏 {game_name} 的评论关键词（好评 vs 差评）")

            # 将评论分为好评和差评
            good_reviews = df_rev[df_rev['voted_up'] == True]['comment'].dropna()
            bad_reviews = df_rev[df_rev['voted_up'] == False]['comment'].dropna()

            # 简单停用词（可扩充）
            stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
            def extract_keywords(texts, top_n=10):
                words = []
                for text in texts:
                    seg_list = jieba.cut(text)
                    for w in seg_list:
                        if len(w) > 1 and w not in stopwords and not w.isdigit():
                            words.append(w)
                counter = Counter(words)
                return counter.most_common(top_n)

            print("\n好评关键词 TOP10：")
            good_keywords = extract_keywords(good_reviews)
            for word, freq in good_keywords:
                print(f"  {word}: {freq}")

            print("\n差评关键词 TOP10：")
            bad_keywords = extract_keywords(bad_reviews)
            for word, freq in bad_keywords:
                print(f"  {word}: {freq}")

            print("\n注：以上仅为关键词统计，可用于初步归因。更深入的 LDA 主题建模等可根据需要扩展。")
    except ImportError:
        print("未安装 jieba 库，跳过 NLP 分析。如需使用请运行：pip install jieba")

    print("\n" + "=" * 60)
    print("所有任务执行完毕！")
    print("=" * 60)

if __name__ == "__main__":
    main()

