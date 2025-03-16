from datasets import load_dataset
import random
import json
import os

def load_all_datasets():
    """加载所有任务的数据集"""
    print("正在加载数据集...")
    datasets = {
        "task1": load_dataset("./datasets/Dolphin_task1"),
        "task2": load_dataset("./datasets/Dolphin_task2"),
        "task3": load_dataset("./datasets/Dolphin_task3")
    }
    return datasets

def merge_datasets(datasets):
    """合并所有任务的数据"""
    print("正在合并数据集...")
    all_data = []
    
    # 从每个数据集的训练集中提取数据
    for task_name, dataset in datasets.items():
        train_data = dataset['train']
        for item in train_data:
            # 添加任务标识
            item['task_type'] = task_name
            all_data.append(item)
    
    return all_data

def split_and_shuffle(data, valid_size=500, seed=42):
    """分割数据集并打乱"""
    print(f"正在分割数据集，验证集大小: {valid_size}")
    random.seed(seed)
    random.shuffle(data)
    
    # 分割验证集和训练集
    valid_data = data[:valid_size]
    train_data = data[valid_size:]
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(valid_data)}")
    
    return train_data, valid_data

def save_to_file(data, filename):
    """保存数据到文件"""
    print(f"正在保存数据到: {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            # 将每条数据转换为JSON字符串并写入文件
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # 设置输出目录
    output_dir = "./processed_data"
    
    # 加载数据集
    datasets = load_all_datasets()
    
    # 合并数据集
    all_data = merge_datasets(datasets)
    
    # 分割并打乱数据
    train_data, valid_data = split_and_shuffle(all_data, valid_size=500)
    
    # 保存处理后的数据
    save_to_file(train_data, f"{output_dir}/train.jsonl")
    save_to_file(valid_data, f"{output_dir}/valid.jsonl")
    
    print("数据处理完成！")
    print(f"训练数据保存在: {output_dir}/train.jsonl")
    print(f"验证数据保存在: {output_dir}/valid.jsonl")

if __name__ == "__main__":
    main()