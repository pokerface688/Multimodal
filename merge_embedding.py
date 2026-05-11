import torch
import os
import glob
import argparse

def merge_embedding_files(root_dir='./embed', pattern='*_embedding.pt', out_file='type_embeddings.pt'):
    """
    将符合 pattern 的所有 *_embedding.pt 文件合并为一个字典：
        key   -> 文件名中的 dataset 部分
        value -> 对应的 embedding 表（torch.Tensor）
    
    Args:
        root_dir (str): 文件路径，默认 './embed/skipgram/'
        pattern (str): 匹配规则，默认 '*_embedding.pt'
        out_file (str): 输出文件名，默认 'type_embedding.pt'
    """
    merged = {}
    
    # 找到所有符合规则的文件
    pattern = os.path.join(root_dir, pattern)
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'未找到任何符合 {pattern} 的文件')
    
    for fp in files:
        # 提取 dataset 名称：去掉后缀 _embedding.pt
        basename = os.path.basename(fp)          # e.g. 'abc_embedding.pt'
        dataset = basename.split('_')[0]
        
        # 加载嵌入表
        emb = torch.load(fp, map_location='cpu')  # 可根据需要调整 map_location
        merged[dataset] = emb
    
    # 保存合并后的字典
    out_file = os.path.join(root_dir, out_file)
    torch.save(merged, out_file)
    print(f'已合并 {len(files)} 个文件 → {out_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./embed')
    args = parser.parse_args()

    merge_embedding_files(args.root_dir)