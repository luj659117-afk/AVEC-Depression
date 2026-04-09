import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os

from train_depression import AVEC2014Dataset, UltimateDepressionModelV2, parse_labels


# ================= 1. 测试集推理核心逻辑 =================
def evaluate_test_set():
    # 测试集只需要单卡即可，50个样本瞬间跑完
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用 {device} 进行测试集评估...")

    # 1. 解析测试集标签
    test_metadata = parse_labels(split='test')
    
    if not test_metadata:
        print('致命错误: 未找到测试集标签，请检查 parse_labels 是否支持 test 划分。')
        return

    feature_dir = 'features_vggface' # 你的特征目录
    test_dataset = AVEC2014Dataset(test_metadata, feature_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 实例化模型并加载最优权重
    model = UltimateDepressionModelV2().to(device)
    weight_path = 'checkpoints/best_vggface_model.pth' # 之前跑出 9.28 时的权重文件
    
    if not os.path.exists(weight_path):
        print(f"找不到权重文件 {weight_path}！")
        return
        
    print(f"加载模型权重: {weight_path}")
    # 因为保存时已经去掉了 'module.' 前缀，所以可以直接 load_state_dict
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 3. 运行推理
    results = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # 获取 batch 中对应的受试者 ID
            batch_size_current = batch[0].size(0)
            start_idx = i * 8 # DataLoader batch_size=8
            batch_subjects = test_dataset.subjects[start_idx : start_idx + batch_size_current]
            
            ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask, labels = [b.to(device) for b in batch]
            
            preds = model(ff_rgb, ff_flow, ff_mask, nw_rgb, nw_flow, nw_mask).squeeze(-1)
            labels = labels.squeeze(-1)
            
            for sub, p, t in zip(batch_subjects, preds.cpu().numpy(), labels.cpu().numpy()):
                results.append({'Subject_ID': sub, 'True_BDI': t, 'Pred_BDI': p, 'Abs_Error': abs(t - p)})
                all_preds.append(p)
                all_labels.append(t)

    # 4. 计算最终指标
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
    
    print("="*40)
    print(f"【纯视觉最终成绩】")
    print(f"Test MAE : {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print("="*40)

    # 5. 导出预测结果用于论文分析
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Abs_Error', ascending=False)
    csv_path = 'vision_only_test_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"预测明细已保存至: {csv_path}")
    print("请查看排名前几的极端错误样本（Abs_Error最大），它们暴露了视觉模型的物理盲区。")

if __name__ == '__main__':
    evaluate_test_set()