import argparse

from runner import Runner
from config import Config

data_config_path = './config/dataset.yaml'
model_config_path = './config/model.yaml'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_config_path', type=str, required=False, default=data_config_path,
                        help='Path of dataset configuration yaml.')

    parser.add_argument('-d', '--dataset', nargs='+', required=True,
                        help='List of activated dataset name.')
    parser.add_argument('--use_image', action='store_true', default=False, 
                        help='是否使用图片编码器，开启则加载图片并编码')
    parser.add_argument('--use_text',action='store_true',default=False, 
                        help='是否使用文本编码器，开启则加载文本并编码')
    parser.add_argument('--use_skipgram', action='store_true', default=False,
                        help='是否使用 skipgram 嵌入表（需 skipgram_embeddings_path）')
    parser.add_argument('--skipgram_embeddings_path', type=str, default=None,
                        help='skipgram 嵌入字典 .pt 路径（与 type_embeddings 同结构）')
    parser.add_argument('--model_config_path', type=str, required=False, default=model_config_path,
                        help='Path of model configuration yaml.')

    parser.add_argument('--seed', type=int, required=False, default=1037,
                        help='Random seed in the config file.')
    
    parser.add_argument('--gpu', type=int, required=False, default=0,
                        help='GPU id in the config file.')

    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--evaluate_only', action='store_true')

    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--opt_lr', type=float, default=None)
    parser.add_argument('--lora_lr', type=float, default=None)
    parser.add_argument('--train_subset_ratio', type=float, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--peft_type', type=str, default=None)
    parser.add_argument('--use_mixlognormal', action='store_true')
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--type_embeddings_path', type=str, default=None)
    parser.add_argument('--time_scale', type=int, default=None)
    parser.add_argument('--loss_ratio', type=float, default=None)
    parser.add_argument('--RCA_ratio', type=float, default=None)
    parser.add_argument('--JEPA_ratio', type=float, default=None)
    parser.add_argument('--RCA_type', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--tem_enc_type', type=str, default=None)

    args = parser.parse_args()

    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    main()

