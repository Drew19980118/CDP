import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle


class YearlyFinancialDataset(Dataset):
    """
    年度财务评级数据集
    一个样本 = 一家公司一年的4个季度数据 → 预测该年度评级
    """

    def __init__(self, dataframe, feature_columns, target_column='rating_id'):
        """
        Parameters:
        -----------
        dataframe : 整合后的数据（包含季度数据）
        feature_columns : 特征列列表
        target_column : 目标列
        """
        self.dataframe = dataframe.sort_values(['corp_code', 'year', 'quarter'])
        self.feature_columns = feature_columns
        self.target_column = target_column

        # 创建年度样本
        self.samples = self._create_yearly_samples()

        print(f"创建了 {len(self.samples)} 个年度样本")
        print(f"特征维度: {len(feature_columns)}")
        print(f"时间序列长度: 4个季度")
        if len(self.samples) > 0:
            print(f"评级类别数: {dataframe[target_column].nunique()}")

    def _create_yearly_samples(self):
        """创建年度样本：每个公司每年4个季度的数据聚合为一个样本"""
        samples = []

        # 按公司和年份分组
        grouped = self.dataframe.groupby(['corp_code', 'year'])

        for (corp_code, year), group in grouped:
            # 确保有完整的4个季度数据
            if len(group) == 4:
                # 按季度排序
                group = group.sort_values('quarter')

                # 提取4个季度的特征 [4, num_features]
                quarterly_features = group[self.feature_columns].values

                # 目标：该年度的评级（应该4个季度都一样）
                target = group[self.target_column].iloc[0]

                sample = {
                    'features': quarterly_features.astype(np.float32),  # [4, num_features]
                    'target': target,
                    'corp_code': corp_code,
                    'year': year,
                    'industry': group['industry_category_id'].iloc[0],
                    'quarters': group['quarter'].tolist()  # 记录包含的季度
                }
                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 特征: [4, num_features] - 4个季度的数据
        features_tensor = torch.FloatTensor(sample['features'])

        if self.target_column == 'rating_id':
            # 分类任务
            target_tensor = torch.LongTensor([sample['target']])
        else:
            # 回归任务
            target_tensor = torch.FloatTensor([sample['target']])

        return {
            'features': features_tensor,  # [4, num_features]
            'target': target_tensor,  # [1]
            'corp_code': sample['corp_code'],
            'year': sample['year'],
            'industry': sample['industry'],
            'quarters': sample['quarters']  # 用于调试
        }

    def save(self, filepath):
        """保存Dataset到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Dataset已保存到: {filepath}")

    @classmethod
    def load(cls, filepath):
        """从文件加载Dataset"""
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Dataset已从 {filepath} 加载")
        return dataset


def build_dataset_from_ratings(company_path, macro_path, rating_path, dataset_name):
    """
    根据评分文件构建对应的Dataset

    Parameters:
    -----------
    company_path : 公司财务数据路径
    macro_path : 宏观数据路径
    rating_path : 评分数据路径
    dataset_name : 数据集名称（用于输出信息）

    Returns:
    --------
    dataset : YearlyFinancialDataset 或 None（如果没有有效数据）
    """
    print(f"\n{'=' * 50}")
    print(f"构建 {dataset_name} 数据集")
    print(f"{'=' * 50}")

    # 1. 读取数据
    try:
        company_df = pd.read_csv(company_path)
        macro_df = pd.read_csv(macro_path)
        rating_df = pd.read_csv(rating_path)
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return None

    print(f"公司数据: {company_df.shape}")
    print(f"宏观数据: {macro_df.shape}")
    print(f"评分数据: {rating_path} - {rating_df.shape}")

    # 2. 预处理公司数据
    company_df['date'] = pd.to_datetime(company_df['date'])
    company_df['year'] = company_df['date'].dt.year
    company_df['quarter'] = company_df['date'].dt.quarter
    company_df['quarter_str'] = company_df['year'].astype(str) + 'Q' + company_df['quarter'].astype(str)

    # 3. 合并公司数据和宏观数据
    merged_df = company_df.merge(
        macro_df,
        left_on=['quarter_str'],
        right_on=['quarter'],
        how='left',
        suffixes=('', '_macro')
    )

    print(f"合并公司+宏观数据后: {merged_df.shape}")

    # 4. 根据评分数据筛选对应的样本
    # 提取评分数据中的公司和年份组合
    rating_combinations = rating_df[['corp_code', 'year']].drop_duplicates()
    print(f"评分数据中的公司-年份组合: {len(rating_combinations)}")

    # 筛选对应的财务数据
    final_df = merged_df.merge(
        rating_combinations,
        on=['corp_code', 'year'],
        how='inner'
    )

    # 合并评分信息
    final_df = final_df.merge(
        rating_df[['corp_code', 'year', 'rating', 'rating_id']],
        on=['corp_code', 'year'],
        how='left'
    )

    print(f"筛选后数据: {final_df.shape}")

    # 5. 检查数据完整性并统计
    yearly_stats = final_df.groupby(['corp_code', 'year']).size().reset_index(name='quarter_count')
    complete_years = yearly_stats[yearly_stats['quarter_count'] == 4]

    print(f"年度数据完整性:")
    print(f"  总公司-年份组合: {len(yearly_stats)}")
    print(f"  完整年度(4个季度): {len(complete_years)}")
    print(f"  不完整年度: {len(yearly_stats) - len(complete_years)}")

    # 只保留完整年度的数据
    complete_mask = final_df.set_index(['corp_code', 'year']).index.isin(
        complete_years.set_index(['corp_code', 'year']).index
    )
    final_df_complete = final_df[complete_mask].copy()

    if len(final_df_complete) == 0:
        print(f"⚠️ 警告: {dataset_name} 没有完整的年度数据!")
        return None

    print(f"最终用于训练的数据: {len(final_df_complete)} 行")
    print(f"唯一公司数: {final_df_complete['corp_code'].nunique()}")
    print(f"年份范围: {final_df_complete['year'].min()} - {final_df_complete['year'].max()}")

    # 6. 识别特征列
    exclude_cols = [
        'corp_code', 'corp_name', 'industry_category_id', 'date', 'year',
        'quarter', 'quarter_str', 'rating', 'rating_id', 'industry_category',
        'quarter_macro', 'year_macro'
    ]

    feature_columns = [col for col in final_df_complete.columns
                       if col not in exclude_cols
                       and final_df_complete[col].dtype in [np.float64, np.float32, np.int64, np.int32]]

    print(f"\n特征统计:")
    print(f"  总特征数: {len(feature_columns)}")

    # 分类特征
    financial_features = [col for col in feature_columns if
                          not col.endswith('_missing') and not any(x in col for x in ['cpi', 'gdp', 'm2', 'm1', 'm0', 'pmi', 'ppi'])]
    missing_features = [col for col in feature_columns if col.endswith('_missing')]
    macro_features = [col for col in feature_columns if
                      any(x in col for x in ['cpi', 'gdp', 'm2', 'm1', 'm0', 'pmi', 'ppi'])]

    print(f"  财务特征: {len(financial_features)}")
    print(f"  Missing指示器: {len(missing_features)}")
    print(f"  宏观特征: {len(macro_features)}")

    # 7. 创建Dataset
    target_column = 'rating_id'
    dataset = YearlyFinancialDataset(final_df_complete, feature_columns, target_column)

    return dataset


def main():
    """
    主函数：根据三个评分文件构建三个Dataset并保存
    """
    # 基础数据路径
    company_path = "data/corporate_financial_panel_data/corporate_financial_panel_data_2014-2024.csv"
    macro_path = "data/macro_data/macro_data_2014-2024.csv"

    # 三个评分文件路径
    rating_paths = {
        'train': "data/corporate_ratings_data/train_set/train_set_split3.csv",
        'validation': "data/corporate_ratings_data/validation_set/validation_set_split3.csv",
        'test': "data/corporate_ratings_data/test_set/test_set_split3.csv"
    }

    # 输出文件路径
    output_dir = "data/processed_datasets"
    os.makedirs(output_dir, exist_ok=True)

    output_paths = {
        'train': os.path.join(output_dir, "train_dataset_split3.pkl"),
        'validation': os.path.join(output_dir, "validation_dataset_split3.pkl"),
        'test': os.path.join(output_dir, "test_dataset_split3.pkl")
    }

    datasets = {}

    # 为每个评分文件构建Dataset
    for split in ['train', 'validation', 'test']:
        rating_path = rating_paths[split]
        output_path = output_paths[split]

        # 构建Dataset
        dataset = build_dataset_from_ratings(
            company_path, macro_path, rating_path, f"{split}数据集"
        )

        if dataset is not None and len(dataset) > 0:
            # 保存Dataset
            dataset.save(output_path)
            datasets[split] = dataset

            # 显示样本信息
            print(f"\n{split}数据集信息:")
            print(f"  样本数量: {len(dataset)}")
            print(f"  特征维度: {len(dataset.feature_columns)}")
            print(f"  目标变量: {dataset.target_column}")

            # 显示一些样本详情
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  样本特征形状: {sample['features'].shape}")
                print(f"  样本目标: {sample['target'].item()}")
        else:
            print(f"❌ {split}数据集构建失败或为空")
            datasets[split] = None

    # 总结报告
    print(f"\n{'=' * 60}")
    print("数据集构建总结")
    print(f"{'=' * 60}")

    total_samples = 0
    for split, dataset in datasets.items():
        if dataset is not None:
            samples_count = len(dataset)
            total_samples += samples_count
            print(f"{split:12}: {samples_count:4d} 个样本")
        else:
            print(f"{split:12}: 构建失败")

    print(f"{'总计:':12} {total_samples:4d} 个样本")

    # 保存特征列信息（用于模型构建）
    feature_info = {}
    for split, dataset in datasets.items():
        if dataset is not None:
            feature_info = {
                'feature_columns': dataset.feature_columns,
                'feature_dim': len(dataset.feature_columns),
                'target_column': dataset.target_column
            }
            break

    if feature_info:
        feature_info_path = os.path.join(output_dir, "feature_info.pkl")
        with open(feature_info_path, 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"\n特征信息已保存到: {feature_info_path}")
        print(f"特征维度: {feature_info['feature_dim']}")
        print(f"特征列示例: {feature_info['feature_columns'][:5]}...")

    return datasets


def load_datasets():
    """
    加载已保存的Dataset
    """
    output_dir = "data/processed_datasets"
    datasets = {}

    for split in ['train', 'validation', 'test']:
        filepath = os.path.join(output_dir, f"{split}_dataset.pkl")
        try:
            datasets[split] = YearlyFinancialDataset.load(filepath)
        except FileNotFoundError:
            print(f"文件未找到: {filepath}")
            datasets[split] = None

    return datasets


if __name__ == "__main__":
    # 构建并保存Dataset
    datasets = main()

    # 测试加载功能
    # print(f"\n{'=' * 50}")
    # print("测试Dataset加载功能")
    # print(f"{'=' * 50}")
    #
    # loaded_datasets = load_datasets()
    #
    # for split, dataset in loaded_datasets.items():
    #     if dataset is not None:
    #         print(f"{split:12}: 成功加载, {len(dataset)} 个样本")
    #     else:
    #         print(f"{split:12}: 加载失败")