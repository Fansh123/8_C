import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置matplotlib为非交互模式
plt.switch_backend('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 生成模拟评论文本
def generate_review_texts(num_reviews=1000):
    """生成模拟评论文本"""
    # 游戏相关词汇
    positive_words = [
        '好玩', '精彩', '刺激', '流畅', '画面精美', '音效出色', '平衡性好', '更新及时',
        '社区活跃', '操作简单', '耐玩', '有挑战性', '内容丰富', '自由度高', '优化好'
    ]
    
    negative_words = [
        '卡顿', 'bug多', '平衡性差', '更新慢', '外挂多', '服务器差', '收费不合理',
        '内容少', '操作复杂', '画面差', '音效差', '匹配系统差', '客服差', '优化差'
    ]
    
    # 生成评论文本
    reviews = []
    for i in range(num_reviews):
        # 随机生成评论长度
        length = np.random.randint(5, 20)
        # 随机决定情感倾向
        is_positive = np.random.random() > 0.3  # 70%正面评论
        
        if is_positive:
            words = np.random.choice(positive_words, size=length, replace=True)
        else:
            words = np.random.choice(negative_words, size=length, replace=True)
        
        review = ' '.join(words)
        reviews.append({'review_text': review, 'voted_up': is_positive})
    
    return pd.DataFrame(reviews)

# 加载评论数据
def load_review_data():
    """加载评论数据"""
    games = {}
    
    # 为每个游戏生成模拟评论数据
    games['CS:GO'] = generate_review_texts(1000)
    games['Dota 2'] = generate_review_texts(1000)
    games['Team Fortress 2'] = generate_review_texts(1000)
    
    return games

# 文本预处理
def preprocess_text(text):
    """文本预处理"""
    # 简单分词（按空格分割）
    words = text.split()
    
    # 停用词表
    stopwords = set([
        '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
        '自己', '这', '游戏', '玩', '觉得', '但是', '比较', '非常', '可以', '还是', '因为'
    ])
    
    # 过滤停用词
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    
    return ' '.join(filtered_words)

# 1. 关键词提取和分析
def extract_keywords(games):
    """提取关键词并分析"""
    print("\n=== 1. 关键词提取和分析 ===")
    
    for game_name, df in games.items():
        print(f"\n{game_name} 关键词分析")
        
        # 预处理文本
        df['processed_text'] = df['review_text'].apply(preprocess_text)
        
        # 分离正面和负面评论
        positive_reviews = df[df['voted_up']]['processed_text']
        negative_reviews = df[~df['voted_up']]['processed_text']
        
        # TF-IDF关键词提取
        vectorizer = TfidfVectorizer(max_features=20, ngram_range=(1, 2))
        
        # 正面评论关键词
        if len(positive_reviews) > 0:
            tfidf_pos = vectorizer.fit_transform(positive_reviews)
            pos_keywords = sorted(list(zip(vectorizer.get_feature_names_out(), tfidf_pos.sum(axis=0).A1)), 
                                key=lambda x: x[1], reverse=True)[:10]
            print("正面评论关键词:")
            for word, score in pos_keywords:
                print(f"  {word}: {score:.4f}")
        
        # 负面评论关键词
        if len(negative_reviews) > 0:
            tfidf_neg = vectorizer.fit_transform(negative_reviews)
            neg_keywords = sorted(list(zip(vectorizer.get_feature_names_out(), tfidf_neg.sum(axis=0).A1)), 
                                key=lambda x: x[1], reverse=True)[:10]
            print("负面评论关键词:")
            for word, score in neg_keywords:
                print(f"  {word}: {score:.4f}")

# 2. 主题聚类分析
def topic_modeling(games):
    """主题聚类分析"""
    print("\n=== 2. 主题聚类分析 ===")
    
    for game_name, df in games.items():
        print(f"\n{game_name} 主题聚类分析")
        
        # 预处理文本
        df['processed_text'] = df['review_text'].apply(preprocess_text)
        
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['processed_text'])
        
        # LDA主题模型
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)
        
        # 打印主题关键词
        feature_names = vectorizer.get_feature_names_out()
        for i, topic in enumerate(lda.components_):
            top_words = [feature_names[j] for j in topic.argsort()[:-11:-1]]
            print(f"主题 {i+1}: {', '.join(top_words)}")

# 3. 情感分析和事件归因
def sentiment_analysis(games):
    """情感分析和事件归因"""
    print("\n=== 3. 情感分析和事件归因 ===")
    
    for game_name, df in games.items():
        print(f"\n{game_name} 情感分析")
        
        # 计算情感倾向
        positive_count = len(df[df['voted_up']])
        negative_count = len(df[~df['voted_up']])
        total_count = len(df)
        
        positive_rate = positive_count / total_count
        negative_rate = negative_count / total_count
        
        print(f"正面评论比例: {positive_rate:.4f}")
        print(f"负面评论比例: {negative_rate:.4f}")
        
        # 基于关键词的事件归因
        print("\n事件归因分析:")
        
        # 预处理文本
        df['processed_text'] = df['review_text'].apply(preprocess_text)
        
        # 定义事件关键词
        event_keywords = {
            '服务器问题': ['服务器', '卡顿', '延迟', '断线'],
            '平衡性问题': ['平衡', 'imba', '强势', '弱势'],
            '内容更新': ['更新', '新内容', '补丁', 'dlc'],
            '外挂问题': ['外挂', '作弊', 'hack', '脚本'],
            '收费问题': ['收费', '价格', '皮肤', '氪金']
        }
        
        # 统计各事件在负面评论中的出现频率
        negative_reviews = df[~df['voted_up']]['processed_text']
        
        if len(negative_reviews) > 0:
            event_counts = {}
            for event, keywords in event_keywords.items():
                count = 0
                for review in negative_reviews:
                    if any(keyword in review for keyword in keywords):
                        count += 1
                event_counts[event] = count
            
            # 计算事件比例
            event_ratios = {event: count / len(negative_reviews) for event, count in event_counts.items()}
            sorted_events = sorted(event_ratios.items(), key=lambda x: x[1], reverse=True)
            
            print("负面评论主要原因:")
            for event, ratio in sorted_events:
                print(f"  {event}: {ratio:.4f}")

# 4. 综合分析与可视化
def comprehensive_analysis(games):
    """综合分析与可视化"""
    print("\n=== 4. 综合分析与可视化 ===")
    
    # 1. 游戏情感对比
    plt.figure(figsize=(12, 6))
    
    game_names = []
    positive_rates = []
    negative_rates = []
    
    for game_name, df in games.items():
        game_names.append(game_name)
        positive_rate = len(df[df['voted_up']]) / len(df)
        negative_rate = 1 - positive_rate
        positive_rates.append(positive_rate)
        negative_rates.append(negative_rate)
    
    x = np.arange(len(game_names))
    width = 0.35
    
    plt.bar(x - width/2, positive_rates, width, label='正面评论')
    plt.bar(x + width/2, negative_rates, width, label='负面评论')
    plt.xticks(x, game_names)
    plt.title('游戏情感倾向对比')
    plt.ylabel('比例')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('game_sentiment_comparison.png')
    plt.close()
    print("游戏情感倾向对比图已保存为 game_sentiment_comparison.png")

# 5. 对游戏厂商的建议
def generate_recommendations():
    """生成对游戏厂商的建议"""
    print("\n=== 5. 对游戏厂商的建议 ===")
    
    recommendations = [
        "1. 重视服务器稳定性和网络优化，减少玩家卡顿和断线问题",
        "2. 定期更新游戏内容，保持玩家新鲜感和游戏活力",
        "3. 建立有效的反外挂机制，维护游戏公平性",
        "4. 关注游戏平衡性，定期调整游戏机制",
        "5. 优化游戏收费模式，确保玩家感受到价值",
        "6. 建立完善的玩家反馈机制，及时响应玩家诉求",
        "7. 加强游戏社区建设，促进玩家互动和归属感",
        "8. 持续优化游戏性能，确保在不同配置设备上的流畅运行"
    ]
    
    for rec in recommendations:
        print(rec)

# 主函数
def main():
    print("开始执行Bonus 1任务：基于自然语言处理的事件归因分析")
    
    # 加载评论数据
    games = load_review_data()
    
    # 1. 关键词提取和分析
    extract_keywords(games)
    
    # 2. 主题聚类分析
    topic_modeling(games)
    
    # 3. 情感分析和事件归因
    sentiment_analysis(games)
    
    # 4. 综合分析与可视化
    comprehensive_analysis(games)
    
    # 5. 对游戏厂商的建议
    generate_recommendations()
    
    print("\nBonus 1任务完成！")

if __name__ == "__main__":
    main()