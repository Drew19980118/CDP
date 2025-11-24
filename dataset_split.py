import pandas as pd
import numpy as np

# 加载数据
ratings_df = pd.read_csv('data/corporate_ratings_data/corporate_ratings_2014-2024.csv')
industries_df = pd.read_csv('data/corporate_industries_data/corporate_industries.csv')

# 合并数据
merged_df = pd.merge(ratings_df, industries_df, on='corp_code', how='inner')
assert len(merged_df) == 14789, "合并后数据量不匹配"

# 创建分层标识
merged_df['strata'] = merged_df['rating_category'].astype(str) + '_' + merged_df['industry_category'].astype(str)


def create_record_level_splits(merged_df, n_splits=3, random_seeds=[42, 123, 456]):
    """
    按记录级别进行三重划分
    """
    all_splits = []

    for split_idx in range(n_splits):
        print(f"正在创建第 {split_idx + 1} 组划分...")

        train_dfs, val_dfs, test_dfs = [], [], []

        for strata in merged_df['strata'].unique():
            subset = merged_df[merged_df['strata'] == strata].copy()

            # 设置随机种子
            base_seed = random_seeds[split_idx]
            strata_seed = hash(strata) % (2 ** 32) + base_seed
            np.random.seed(strata_seed)

            # 获取该分层所有记录的索引
            record_indices = subset.index.values
            n_records = len(record_indices)

            if n_records == 0:
                continue

            # 计算各集合的记录数量
            n_train = max(1, round(n_records * 0.75))
            n_val = max(1, round(n_records * 0.15))
            n_test = max(1, n_records - n_train - n_val)

            # 随机打乱记录索引
            shuffled_indices = np.random.permutation(record_indices)

            # 按记录划分
            train_indices = shuffled_indices[:n_train]
            val_indices = shuffled_indices[n_train:n_train + n_val]
            test_indices = shuffled_indices[n_train + n_val:n_train + n_val + n_test]

            # 分配到各集合
            train_dfs.append(merged_df.loc[train_indices])
            val_dfs.append(merged_df.loc[val_indices])
            test_dfs.append(merged_df.loc[test_indices])

        # 合并结果
        train_split = pd.concat(train_dfs)
        val_split = pd.concat(val_dfs)
        test_split = pd.concat(test_dfs)

        all_splits.append({
            'train': train_split,
            'val': val_split,
            'test': test_split,
            'split_id': split_idx
        })

    return all_splits


def analyze_strata_distribution(all_splits, merged_df):
    """
    详细分析每个分层在train/val/test中的记录分布
    """
    for split_idx, split in enumerate(all_splits):
        print(f"\n{'=' * 80}")
        print(f"第 {split_idx + 1} 组划分 - 各分层记录分布")
        print(f"{'=' * 80}")

        # 获取当前划分的三个集合
        train = split['train']
        val = split['val']
        test = split['test']

        # 创建统计表格
        stats_data = []

        for strata in merged_df['strata'].unique():
            # 原始数据
            orig_count = len(merged_df[merged_df['strata'] == strata])

            # 各集合中的数据
            train_count = len(train[train['strata'] == strata])
            val_count = len(val[val['strata'] == strata])
            test_count = len(test[test['strata'] == strata])

            # 计算比例
            if orig_count > 0:
                train_ratio = train_count / orig_count
                val_ratio = val_count / orig_count
                test_ratio = test_count / orig_count
            else:
                train_ratio = val_ratio = test_ratio = 0

            stats_data.append({
                'strata': strata,
                'original': orig_count,
                'train': train_count,
                'val': val_count,
                'test': test_count,
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio
            })

        # 转换为DataFrame并排序
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.sort_values('original', ascending=False)

        # 打印详细统计
        print(
            f"{'分层':<45} {'原始':>6} {'训练集':>8} {'验证集':>8} {'测试集':>8} {'训练比例':>10} {'验证比例':>10} {'测试比例':>10}")
        print(
            f"{'-' * 45} {'------':>6} {'--------':>8} {'--------':>8} {'--------':>8} {'----------':>10} {'----------':>10} {'----------':>10}")

        for _, row in stats_df.iterrows():
            print(
                f"{row['strata']:<45} {row['original']:>6} {row['train']:>8} {row['val']:>8} {row['test']:>8} {row['train_ratio']:>10.1%} {row['val_ratio']:>10.1%} {row['test_ratio']:>10.1%}")

        # 汇总统计
        total_original = stats_df['original'].sum()
        total_train = stats_df['train'].sum()
        total_val = stats_df['val'].sum()
        total_test = stats_df['test'].sum()

        print(
            f"{'-' * 45} {'------':>6} {'--------':>8} {'--------':>8} {'--------':>8} {'----------':>10} {'----------':>10} {'----------':>10}")
        print(
            f"{'总计':<45} {total_original:>6} {total_train:>8} {total_val:>8} {total_test:>8} {total_train / total_original:>10.1%} {total_val / total_original:>10.1%} {total_test / total_original:>10.1%}")

        # 检查小样本分层的处理
        print(f"\n小样本分层检查 (原始记录数 < 10):")
        small_strata = stats_df[stats_df['original'] < 10]
        if len(small_strata) > 0:
            for _, row in small_strata.iterrows():
                print(
                    f"  {row['strata']:<45} 原始:{row['original']:2d} → 训练:{row['train']:2d} 验证:{row['val']:2d} 测试:{row['test']:2d}")
        else:
            print("  无小样本分层")


def validate_splits_integrity(all_splits):
    """
    验证划分的完整性
    """
    print("=== 划分完整性验证 ===")

    for i, split in enumerate(all_splits):
        train = split['train']
        val = split['val']
        test = split['test']

        # 检查记录是否有重叠
        train_indices = set(train.index)
        val_indices = set(val.index)
        test_indices = set(test.index)

        overlap_train_val = len(train_indices & val_indices)
        overlap_train_test = len(train_indices & test_indices)
        overlap_val_test = len(val_indices & test_indices)

        print(f"第 {i + 1} 组划分重叠检查:")
        print(f"  训练集-验证集重叠: {overlap_train_val} 条记录")
        print(f"  训练集-测试集重叠: {overlap_train_test} 条记录")
        print(f"  验证集-测试集重叠: {overlap_val_test} 条记录")

        # 检查数据完整性
        total_original = len(merged_df)
        total_split = len(train) + len(val) + len(test)
        print(f"  数据完整性: {total_split}/{total_original} ({total_split / total_original:.2%})")
        print()


# # 执行三重划分
print("开始进行三重记录级别划分...")
all_splits = create_record_level_splits(merged_df, n_splits=3, random_seeds=[42, 123, 456])

# 验证划分完整性
validate_splits_integrity(all_splits)

# 详细分析每个分层的分布
analyze_strata_distribution(all_splits, merged_df)