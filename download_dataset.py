import os
import shutil

# ================= 配置区域 =================
# 1. 强制设置环境变量使用 hf-mirror (必须在 import datasets 前设置)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 这是你报错日志中提示的缺失路径
SAVE_PATH = "/newdata/DataSets/wikipedia_zh"

# 3. 使用一个稳定的中文维基百科源
# 原代码用的 wikipedia 20220301 已过期，这里使用 pleisto 清洗版，质量更高且稳定
DATASET_NAME = "pleisto/wikipedia-cn-20230720-filtered"
# ===========================================

import datasets
from datasets import load_dataset

def main():
    print(f"🚀 开始通过 hf-mirror 下载: {DATASET_NAME}")
    print(f"📂 目标保存路径: {SAVE_PATH}")

    # 检查目录是否存在，不存在则创建
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"已创建目录: {SAVE_PATH}")
    else:
        print(f"⚠️ 目录已存在: {SAVE_PATH} (如果脚本运行失败，请手动删除该目录后重试)")

    try:
        # 开始下载
        # split='train' 确保我们拿到的是 Dataset 对象而不是 DatasetDict
        dataset = load_dataset(DATASET_NAME, split='train', trust_remote_code=True)
        
        print(f"✅ 下载成功！数据量: {len(dataset)} 条")
        print("💾 正在转换并保存到本地磁盘 (Save to Disk)...")

        # 核心步骤：保存为 load_from_disk 能读取的格式
        dataset.save_to_disk(SAVE_PATH)

        print("-" * 30)
        print(f"🎉 成功！所有数据已保存至: {SAVE_PATH}")
        print("现在你可以直接运行原本的 pruning 脚本了，它会直接读取本地数据。")
        print("-" * 30)

    except Exception as e:
        print("\n❌ 发生错误:")
        print(e)
        print("\n建议排查：")
        print("1. 确保服务器能访问 https://hf-mirror.com")
        print("2. 检查磁盘空间是否充足")

if __name__ == "__main__":
    main()