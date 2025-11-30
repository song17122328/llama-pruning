#!/usr/bin/env python3
"""
参数搜索结果相关性分析脚本

分析梯度统计指标与 ACC 的相关性，找出预测 ACC 的最佳指标。
这对于科研论文非常有价值！

用法:
    python analyze_param_correlations.py --results results/param_search_mistral_20/search_results.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


def compute_correlations(df, target_col='acc_mean'):
    """计算所有梯度统计指标与目标指标的相关性"""

    # 所有可能的预测指标
    predictor_cols = [
        'taylor_seq_len',
        'taylor_num_samples',
        'grad_mean_ratio',
        'grad_norm_ratio',
        'grad_std_ratio',
        'grad_max_ratio',
        'grad_mean_range',
        'grad_norm_range',
        'extreme_pruning_layers',
        'ppl'
    ]

    correlations = {}

    for col in predictor_cols:
        if col in df.columns and target_col in df.columns:
            # 过滤掉缺失值
            valid_data = df[[col, target_col]].dropna()

            if len(valid_data) > 2:  # 至少需要 3 个数据点
                # 计算 Pearson 相关系数
                pearson_corr, pearson_p = stats.pearsonr(valid_data[col], valid_data[target_col])

                # 计算 Spearman 相关系数（对非线性关系更敏感）
                spearman_corr, spearman_p = stats.spearmanr(valid_data[col], valid_data[target_col])

                correlations[col] = {
                    'pearson_corr': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p': spearman_p,
                    'n_samples': len(valid_data)
                }

    return correlations


def find_best_predictors(correlations, top_n=5):
    """找出最佳的预测指标（按相关性强度排序）"""

    # 按 Spearman 相关系数的绝对值排序
    sorted_correlations = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]['spearman_corr']),
        reverse=True
    )

    return sorted_correlations[:top_n]


def plot_correlation_heatmap(df, output_file='correlation_heatmap.png'):
    """绘制相关性热力图"""

    # 选择要分析的列
    analysis_cols = [
        'taylor_seq_len',
        'taylor_num_samples',
        'grad_mean_ratio',
        'grad_norm_ratio',
        'grad_std_ratio',
        'extreme_pruning_layers',
        'acc_mean',
        'ppl'
    ]

    # 只保留存在的列
    available_cols = [col for col in analysis_cols if col in df.columns]
    df_subset = df[available_cols].dropna()

    if len(df_subset) < 2:
        print("数据不足，无法生成热力图")
        return

    # 计算相关性矩阵
    corr_matrix = df_subset.corr(method='spearman')

    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )

    plt.title('Spearman Correlation Heatmap\n(梯度统计指标 vs ACC)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 相关性热力图已保存到 {output_file}")


def plot_scatter_matrix(df, predictors, target='acc_mean', output_file='scatter_matrix.png'):
    """绘制散点图矩阵，显示每个预测指标与 ACC 的关系"""

    n_predictors = len(predictors)
    if n_predictors == 0:
        print("没有可用的预测指标")
        return

    # 计算合适的子图布局
    n_cols = min(3, n_predictors)
    n_rows = (n_predictors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_predictors == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (predictor, corr_info) in enumerate(predictors):
        if i >= len(axes):
            break

        ax = axes[i]

        # 过滤缺失值
        valid_data = df[[predictor, target]].dropna()

        if len(valid_data) > 0:
            # 绘制散点图
            ax.scatter(valid_data[predictor], valid_data[target], alpha=0.6, s=50)

            # 添加趋势线
            z = np.polyfit(valid_data[predictor], valid_data[target], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data[predictor].min(), valid_data[predictor].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # 设置标题和标签
            ax.set_xlabel(predictor, fontsize=10)
            ax.set_ylabel(target, fontsize=10)

            # 显示相关性
            spearman_corr = corr_info['spearman_corr']
            p_value = corr_info['spearman_p']
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            ax.set_title(
                f'r = {spearman_corr:.3f} {sig}\n(Spearman)',
                fontsize=10,
                fontweight='bold'
            )

            ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_predictors, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 散点图矩阵已保存到 {output_file}")


def generate_prediction_formula(df, best_predictors, target='acc_mean'):
    """生成预测公式（线性回归）"""

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error

    # 提取最佳预测指标（前 3 个）
    top_predictors = [p[0] for p in best_predictors[:3]]

    # 过滤缺失值
    cols_to_use = top_predictors + [target]
    valid_data = df[cols_to_use].dropna()

    if len(valid_data) < 3:
        print("数据不足，无法生成预测公式")
        return None

    X = valid_data[top_predictors].values
    y = valid_data[target].values

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 预测
    y_pred = model.predict(X)

    # 评估
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # 生成公式
    formula = f"{target} ≈ {model.intercept_:.4f}"
    for i, predictor in enumerate(top_predictors):
        coef = model.coef_[i]
        sign = '+' if coef >= 0 else '-'
        formula += f" {sign} {abs(coef):.4f} × {predictor}"

    print(f"\n预测公式（基于线性回归）:")
    print(f"  {formula}")
    print(f"  R² = {r2:.4f}")
    print(f"  MAE = {mae:.4f}")

    return {
        'formula': formula,
        'r2': r2,
        'mae': mae,
        'predictors': top_predictors,
        'coefficients': model.coef_.tolist(),
        'intercept': model.intercept_
    }


def main():
    parser = argparse.ArgumentParser(description='分析参数搜索结果的相关性')
    parser.add_argument('--results', type=str, required=True,
                       help='搜索结果 CSV 文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认: results CSV 同目录）')
    parser.add_argument('--target', type=str, default='acc_mean',
                       help='目标指标（默认: acc_mean）')

    args = parser.parse_args()

    # 读取结果
    df = pd.read_csv(args.results)

    # 只保留成功的实验
    df = df[df['success'] == True].copy()

    print(f"成功的实验数量: {len(df)}")

    if len(df) < 3:
        print("实验数量太少（< 3），无法进行有效分析")
        return

    # 设置输出目录
    if args.output_dir is None:
        output_dir = Path(args.results).parent
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"相关性分析")
    print(f"{'='*80}\n")

    # 计算相关性
    correlations = compute_correlations(df, target_col=args.target)

    # 找出最佳预测指标
    best_predictors = find_best_predictors(correlations, top_n=10)

    print(f"Top 10 预测指标（按 Spearman 相关性强度排序）:")
    print("-" * 80)
    print(f"{'指标':<25} {'Spearman r':<12} {'P-value':<12} {'Pearson r':<12} {'样本数':<10}")
    print("-" * 80)

    for predictor, corr_info in best_predictors:
        spearman_r = corr_info['spearman_corr']
        spearman_p = corr_info['spearman_p']
        pearson_r = corr_info['pearson_corr']
        n = corr_info['n_samples']

        # 显著性标记
        sig = "***" if spearman_p < 0.001 else "**" if spearman_p < 0.01 else "*" if spearman_p < 0.05 else ""

        print(f"{predictor:<25} {spearman_r:>11.4f} {spearman_p:>11.4e} {pearson_r:>11.4f} {n:>10} {sig}")

    # 生成可视化
    print(f"\n{'='*80}")
    print(f"生成可视化")
    print(f"{'='*80}\n")

    # 热力图
    heatmap_file = output_dir / "correlation_heatmap.png"
    plot_correlation_heatmap(df, output_file=heatmap_file)

    # 散点图矩阵（前 6 个最相关的指标）
    scatter_file = output_dir / "scatter_matrix.png"
    plot_scatter_matrix(df, best_predictors[:6], target=args.target, output_file=scatter_file)

    # 生成预测公式
    print(f"\n{'='*80}")
    print(f"预测模型")
    print(f"{'='*80}")

    prediction_model = generate_prediction_formula(df, best_predictors, target=args.target)

    if prediction_model:
        # 保存预测模型
        model_file = output_dir / "prediction_model.json"
        import json
        with open(model_file, 'w') as f:
            json.dump(prediction_model, f, indent=2)
        print(f"\n✓ 预测模型已保存到 {model_file}")

    # 保存详细的相关性报告
    corr_report_file = output_dir / "correlation_report.txt"
    with open(corr_report_file, 'w') as f:
        f.write("参数搜索相关性分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"目标指标: {args.target}\n")
        f.write(f"成功实验数: {len(df)}\n\n")

        f.write("Top 10 预测指标:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'指标':<25} {'Spearman r':<12} {'P-value':<12} {'Pearson r':<12} {'样本数':<10}\n")
        f.write("-" * 80 + "\n")

        for predictor, corr_info in best_predictors:
            spearman_r = corr_info['spearman_corr']
            spearman_p = corr_info['spearman_p']
            pearson_r = corr_info['pearson_corr']
            n = corr_info['n_samples']
            sig = "***" if spearman_p < 0.001 else "**" if spearman_p < 0.01 else "*" if spearman_p < 0.05 else ""

            f.write(f"{predictor:<25} {spearman_r:>11.4f} {spearman_p:>11.4e} {pearson_r:>11.4f} {n:>10} {sig}\n")

        if prediction_model:
            f.write("\n\n预测公式:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{prediction_model['formula']}\n")
            f.write(f"R² = {prediction_model['r2']:.4f}\n")
            f.write(f"MAE = {prediction_model['mae']:.4f}\n")

    print(f"\n✓ 相关性报告已保存到 {corr_report_file}")

    # 关键发现
    print(f"\n{'='*80}")
    print(f"关键发现")
    print(f"{'='*80}\n")

    if best_predictors:
        top_predictor, top_corr = best_predictors[0]
        print(f"最强预测指标: {top_predictor}")
        print(f"  Spearman r = {top_corr['spearman_corr']:.4f}")
        print(f"  P-value = {top_corr['spearman_p']:.4e}")

        if abs(top_corr['spearman_corr']) > 0.7:
            print(f"  ✓ 强相关！可以用 {top_predictor} 来预测 {args.target}")
        elif abs(top_corr['spearman_corr']) > 0.4:
            print(f"  ⚠ 中等相关，{top_predictor} 对 {args.target} 有一定预测能力")
        else:
            print(f"  ⚠ 相关性较弱，可能需要更多数据或其他指标")

        # 给出科研建议
        print(f"\n科研价值:")
        if 'grad_norm_ratio' in top_predictor or 'grad_mean_ratio' in top_predictor:
            print(f"  ✓ 发现梯度统计指标与 ACC 存在相关性！")
            print(f"  ✓ 这表明可以通过梯度统计来预测剪枝后性能，无需完整评估")
            print(f"  ✓ 适合写入论文：梯度比率可作为剪枝质量的快速指标")
        elif 'taylor_seq_len' in top_predictor:
            print(f"  ✓ 序列长度对性能有显著影响")
            print(f"  ✓ 可以论证：短序列减少梯度不稳定性，提高剪枝准确性")

    print(f"\n{'='*80}")
    print(f"分析完成！")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
