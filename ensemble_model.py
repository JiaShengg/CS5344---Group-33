# ensemble_model.py
# 贷款违约预测 - 多模型集成版本

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import warnings
import time
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# 1. 加载数据
def load_data():
    train_df = pd.read_csv('data/loans_train.csv')
    valid_df = pd.read_csv('data/loans_valid.csv')
    test_df = pd.read_csv('data/loans_test.csv')
    return train_df, valid_df, test_df

# 2. 计算每月还款额
def calculate_monthly_payment(principal, annual_rate, term_months):
    # 将年利率转换为月利率
    monthly_rate = annual_rate / 100 / 12
    
    # 计算等额本息每月还款额
    if monthly_rate == 0:
        return principal / term_months
    return principal * monthly_rate * (1 + monthly_rate) ** term_months / ((1 + monthly_rate) ** term_months - 1)

# 3. 计算"实付/应付"比率，并使用滑动窗口统计值
def calculate_payment_ratios(df):
    payment_ratios = pd.DataFrame(index=df.index)
    
    # 先计算每期的比率
    raw_ratios = []
    for period in range(1, 14):  # 从第1期开始，因为第0期没有之前的数据可比较
        # 获取本期和上期的余额
        current_upb_col = f"{period}_CurrentActualUPB"
        prev_upb_col = f"{period-1}_CurrentActualUPB"
        
        # 计算实际支付金额 = 上期余额 - 本期余额（如果是正数）
        actual_payment = df[prev_upb_col] - df[current_upb_col]
        actual_payment = actual_payment.clip(lower=0)  # 确保不为负
        
        # 计算应付金额（基于原始贷款金额、利率、期限）
        scheduled_payment = df.apply(
            lambda x: calculate_monthly_payment(
                x['OriginalUPB'], 
                x['OriginalInterestRate'], 
                x['OriginalLoanTerm']
            ), 
            axis=1
        )
        
        # 计算比率（避免除以零）
        ratio = actual_payment / scheduled_payment.replace(0, np.nan)
        
        # 处理异常值：对99%以上的异常值进行截断
        upper_bound = ratio.quantile(0.99)
        ratio = ratio.clip(upper=upper_bound)
        
        payment_ratios[f"payment_ratio_{period}"] = ratio
        raw_ratios.append(ratio)
    
    # 计算滑动窗口统计值（窗口大小为3）
    for i in range(3, 14):  # 从第3期开始，可以计算3期窗口的统计值
        # 计算最近3期的均值
        payment_ratios[f"ratio_mean_win3_{i}"] = payment_ratios[[f"payment_ratio_{i-2}", 
                                                               f"payment_ratio_{i-1}", 
                                                               f"payment_ratio_{i}"]].mean(axis=1)
        
        # 计算最近3期的标准差（反映波动性）
        payment_ratios[f"ratio_std_win3_{i}"] = payment_ratios[[f"payment_ratio_{i-2}", 
                                                              f"payment_ratio_{i-1}", 
                                                              f"payment_ratio_{i}"]].std(axis=1)
        
        # 计算最近3期的趋势（简单的线性趋势）
        payment_ratios[f"ratio_trend_win3_{i}"] = (payment_ratios[f"payment_ratio_{i}"] - 
                                                 payment_ratios[f"payment_ratio_{i-2}"]) / 2
    
    # 计算总体统计值
    payment_ratios['avg_payment_ratio'] = payment_ratios[[col for col in payment_ratios.columns 
                                                        if col.startswith('payment_ratio_')]].mean(axis=1)
    payment_ratios['std_payment_ratio'] = payment_ratios[[col for col in payment_ratios.columns 
                                                        if col.startswith('payment_ratio_')]].std(axis=1)
    payment_ratios['min_payment_ratio'] = payment_ratios[[col for col in payment_ratios.columns 
                                                        if col.startswith('payment_ratio_')]].min(axis=1)
    payment_ratios['max_payment_ratio'] = payment_ratios[[col for col in payment_ratios.columns 
                                                        if col.startswith('payment_ratio_')]].max(axis=1)
    
    # 计算最近3期的均值（作为一个重要特征）
    payment_ratios['recent3_avg_ratio'] = payment_ratios[[f"payment_ratio_{i}" for i in range(11, 14)]].mean(axis=1)
    
    # 添加一些差分特征（相邻月份的变化）
    for i in range(2, 14):
        payment_ratios[f"ratio_diff_{i}"] = payment_ratios[f"payment_ratio_{i}"] - payment_ratios[f"payment_ratio_{i-1}"]
    
    # 计算低于1.0的比率数量（表示未完全还款）
    under_payment_cols = [col for col in payment_ratios.columns if col.startswith('payment_ratio_')]
    payment_ratios['under_payment_count'] = (payment_ratios[under_payment_cols] < 1.0).sum(axis=1)
    
    # 计算为0的比率数量（表示没有还款）
    payment_ratios['zero_payment_count'] = (payment_ratios[under_payment_cols] == 0).sum(axis=1)
    
    return payment_ratios

# 4. 构建特征集
def build_features(df, is_train=True):
    # 计算还款比率特征
    payment_features = calculate_payment_ratios(df)
    
    # 处理缺失值和异常值
    numeric_cols = payment_features.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        # 处理异常值
        for col in numeric_cols:
            if payment_features[col].notna().sum() > 0:
                upper_bound = payment_features[col].quantile(0.99)
                lower_bound = payment_features[col].quantile(0.01)
                payment_features[col] = payment_features[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 填充缺失值
        numeric_imputer = SimpleImputer(strategy='median')
        payment_features[numeric_cols] = numeric_imputer.fit_transform(payment_features[numeric_cols])
    
    # 标准化特征
    scaler = StandardScaler()
    if is_train:
        payment_features_scaled = pd.DataFrame(
            scaler.fit_transform(payment_features),
            columns=payment_features.columns,
            index=payment_features.index
        )
        # 保存scaler供后续使用
        globals()['_scaler'] = scaler
    else:
        # 使用训练集上拟合的scaler
        if '_scaler' in globals():
            scaler = globals()['_scaler']
            payment_features_scaled = pd.DataFrame(
                scaler.transform(payment_features),
                columns=payment_features.columns,
                index=payment_features.index
            )
        else:
            # 如果没有保存的scaler，则不做标准化
            payment_features_scaled = payment_features.copy()
    
    return payment_features_scaled

# 5. 训练基本模型
def train_base_models(X_train, contamination=None):
    models = {}
    
    # 设置contamination参数
    if contamination is None:
        contamination_value = 'auto'
    else:
        contamination_value = contamination
    
    # 1. IsolationForest模型
    models['isolation_forest'] = IsolationForest(
        n_estimators=100, 
        contamination=contamination_value,
        random_state=42,
        n_jobs=-1
    )
    
    # 2. 带不同参数的IsolationForest
    models['isolation_forest_more_trees'] = IsolationForest(
        n_estimators=200,
        max_samples='auto',
        contamination=contamination_value,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. One-Class SVM
    models['one_class_svm'] = OneClassSVM(
        kernel='rbf',
        gamma='auto',
        # nu参数保留默认值0.5 (对于OC-SVM，nu近似等于contamination)
        nu=0.5 if contamination_value == 'auto' else min(contamination_value, 0.5),
        shrinking=True
    )
    
    # 4. 椭圆包络 (对异常敏感的稳健协方差估计)
    models['robust_covariance'] = EllipticEnvelope(
        contamination=contamination_value,
        random_state=42
    )
    
    # 5. 局部离群因子
    models['lof'] = LocalOutlierFactor(
        n_neighbors=35,
        contamination=contamination_value,
        novelty=True,
        n_jobs=-1
    )
    
    
    # 训练模型
    for name, model in models.items():
        model.fit(X_train)
    
    return models

# 6. 生成各个模型的预测分数
def generate_predictions(models, X):
    
    predictions = {}
    
    for name, model in models.items():
        if name in ['isolation_forest', 'isolation_forest_more_trees', 'one_class_svm', 'robust_covariance', 'lof']:
            # 对于异常检测模型，使用决策函数值的负数作为异常分数（越大越异常）
            scores = -model.decision_function(X)
            predictions[name] = scores
        elif name == 'kmeans':
            # 对于KMeans，使用样本到最近聚类中心的距离作为异常分数
            centers = model.cluster_centers_
            # 计算每个样本到所有中心的距离
            distances = np.zeros((X.shape[0], centers.shape[0]))
            for i, center in enumerate(centers):
                distances[:, i] = np.linalg.norm(X - center, axis=1)
            # 使用到最近中心的距离作为异常分数
            scores = np.min(distances, axis=1)
            predictions[name] = scores
    
    return predictions

# 7. 使用加权集成方法
def ensemble_predictions(predictions, weights=None):
    if weights is None:
        # 默认情况下平均权重
        weights = {name: 1/len(predictions) for name in predictions}
    
    # 标准化各个模型的分数
    normalized_preds = {}
    for name, scores in predictions.items():
        if len(scores) > 1:
            # 使用百分比排名进行标准化
            ranks = stats.rankdata(scores)
            normalized = ranks / len(ranks)
            normalized_preds[name] = normalized
        else:
            normalized_preds[name] = scores
    
    # 加权平均
    ensemble_scores = np.zeros(len(next(iter(normalized_preds.values()))))
    for name, scores in normalized_preds.items():
        ensemble_scores += weights[name] * scores
    
    # 将结果缩放到[0, 1]
    if ensemble_scores.max() > ensemble_scores.min():
        ensemble_scores = (ensemble_scores - ensemble_scores.min()) / (ensemble_scores.max() - ensemble_scores.min())
    
    return ensemble_scores

# 8. 优化集成权重
def optimize_weights(predictions, y_true):
    # 使用网格搜索找到最优权重组合
    best_ap = 0
    best_weights = {name: 1/len(predictions) for name in predictions}
    
    # 对于每个模型，尝试不同的权重
    weight_options = np.arange(0.1, 1.1, 0.1)
    model_names = list(predictions.keys())
    
    if len(model_names) == 2:
        # 如果只有两个模型，那么可以尝试所有权重组合
        for w1 in weight_options:
            w2 = 1 - w1
            weights = {model_names[0]: w1, model_names[1]: w2}
            
            ensemble_scores = ensemble_predictions(predictions, weights)
            ap = average_precision_score(y_true, ensemble_scores)
            
            if ap > best_ap:
                best_ap = ap
                best_weights = weights.copy()
    
    elif len(model_names) > 2:
        # 对于多个模型，我们使用贪婪搜索
        # 从平均权重开始
        current_weights = {name: 1/len(predictions) for name in predictions}
        current_ap = average_precision_score(y_true, ensemble_predictions(predictions, current_weights))
        
        improved = True
        while improved:
            improved = False
            
            for model in model_names:
                for weight in weight_options:
                    # 尝试为当前模型设置一个新权重
                    test_weights = current_weights.copy()
                    old_weight = test_weights[model]
                    test_weights[model] = weight
                    
                    # 归一化权重使其总和为1
                    weight_sum = sum(test_weights.values())
                    for m in model_names:
                        test_weights[m] /= weight_sum
                    
                    ensemble_scores = ensemble_predictions(predictions, test_weights)
                    ap = average_precision_score(y_true, ensemble_scores)
                    
                    
                    if ap > current_ap:
                        current_ap = ap
                        current_weights = test_weights.copy()
                        improved = True
                        best_ap = ap
                        best_weights = test_weights.copy()
                
                if improved:
                    # 如果找到了更好的权重，重新开始循环
                    break
    
    return best_weights

# 9. 应用于测试集
def apply_to_test(models, X_test, best_weights):
    # 生成各个模型的预测
    predictions = generate_predictions(models, X_test)
    
    # 集成预测
    ensemble_scores = ensemble_predictions(predictions, best_weights)
    
    # 使用Beta分布调整概率分布
    # Beta分布参数alpha=beta=0.5会产生类似U型分布，更有区分度
    default_probability = stats.beta.cdf(ensemble_scores, 0.5, 0.5)
    
    return default_probability

# 10. 主函数
def main():
    # 加载数据
    train_df, valid_df, test_df = load_data()
    
    # 构建特征
    X_train = build_features(train_df, is_train=True)
    y_valid = valid_df['target'].copy()  # 使用验证集标签评估模型
    
    # 打印标签分布
    print(f"验证集标签分布：\n{y_valid.value_counts()}")
    
    X_valid = build_features(valid_df, is_train=False)
    X_test = build_features(test_df, is_train=False)
    
    # 定义一组contamination值进行调优
    contamination_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 'auto']
    
    # 存储不同contamination值下的模型和评估结果
    all_models = {}
    all_predictions = {}
    all_scores = {}
    
    # 减少print输出，只显示进度
    print(f"开始测试{len(contamination_values)}个不同的contamination值...")
    
    # 遍历不同的contamination值
    for i, contamination in enumerate(contamination_values):
        print(f"进度: [{i+1}/{len(contamination_values)}] 测试 contamination={contamination}", end="\r")
        
        # 训练基本模型 - 减少内部打印
        models = train_base_models(X_train, contamination)
        all_models[str(contamination)] = models
        
        # 在验证集上生成预测 - 减少内部打印
        valid_predictions = generate_predictions(models, X_valid)
        all_predictions[str(contamination)] = valid_predictions
        
        # 评估各个基本模型 - 不打印详情
        ap_scores = {}
        for name, scores in valid_predictions.items():
            ap = average_precision_score(y_valid, scores)
            ap_scores[name] = ap
        
        # 存储分数
        all_scores[str(contamination)] = ap_scores
        
        # 优化集成权重 - 减少内部打印
        best_weights = optimize_weights(valid_predictions, y_valid)
        
        # 使用最佳权重集成预测
        valid_ensemble = ensemble_predictions(valid_predictions, best_weights)
        valid_ap = average_precision_score(y_valid, valid_ensemble)
        all_scores[str(contamination)]['ensemble'] = valid_ap
    
    # 完成后换行
    print()
    
    # 找出最佳的contamination值和对应的模型
    best_contamination = None
    best_ensemble_ap = 0
    best_model_ap = {}
    
    print("\n===== 不同contamination值的集成模型AP比较 =====")
    for contamination, scores in all_scores.items():
        ensemble_ap = scores.get('ensemble', 0)
        print(f"contamination={contamination}: AP={ensemble_ap:.4f}")
        
        if ensemble_ap > best_ensemble_ap:
            best_ensemble_ap = ensemble_ap
            best_contamination = contamination
            best_model_ap = {k: v for k, v in scores.items() if k != 'ensemble'}
    
    print(f"\n最佳contamination值: {best_contamination}, 集成模型AP: {best_ensemble_ap:.4f}")
    print("\n各基础模型在最佳contamination值下的AP分数:")
    for model_name, ap in best_model_ap.items():
        print(f"{model_name}: {ap:.4f}")
    
    # 使用最佳contamination值的模型和权重
    best_models = all_models[best_contamination]
    best_predictions = all_predictions[best_contamination]
    best_weights = optimize_weights(best_predictions, y_valid)
    
    # 应用于测试集
    test_probs = apply_to_test(best_models, X_test, best_weights)
    
    # 创建提交文件
    results = pd.DataFrame({
        'Id': test_df.index,
        'target': test_probs
    })
    
    # 保存结果
    results.to_csv('ensemble_loan_default_predictions.csv', index=False)
    print(f"结果已保存至 ensemble_loan_default_predictions.csv，共 {len(results)} 条预测")
    
    # 保存调参结果记录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_results = {
        "timestamp": timestamp,
        "contamination_values_tested": [str(c) for c in contamination_values],
        "best_contamination": str(best_contamination),
        "best_ensemble_ap": best_ensemble_ap,
        "model_scores": {str(c): {k: float(v) for k, v in scores.items()} for c, scores in all_scores.items()},
        "best_weights": {k: float(v) for k, v in best_weights.items()}
    }
    
    # 将结果保存为JSON文件
    with open(f"contamination_tuning_results_{timestamp}.json", "w") as f:
        json.dump(experiment_results, f, indent=2)
    
    print(f"调参实验结果已保存至 contamination_tuning_results_{timestamp}.json")
    

if __name__ == "__main__":
    main()
