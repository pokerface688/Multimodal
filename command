#!/bin/bash

## Qwen2-0.5B
# lora finetune
python main.py -d taxi --peft_type lora --regen_embed
python main.py -d taobao --peft_type lora 
python main.py -d amazon --peft_type lora
python main.py -d retweet --peft_type lora

# llm freeze
python main.py -d taxi --peft_type freeze
python main.py -d taobao --peft_type freeze
python main.py -d amazon --peft_type freeze
python main.py -d retweet --peft_type freeze

# --------------------------------------------------------------

# lora finetune + prompt
python main.py -d taxi --peft_type lora --use_prompt
python main.py -d taobao --peft_type lora --use_prompt
python main.py -d amazon --peft_type lora --use_prompt
python main.py -d retweet --peft_type lora --use_prompt

# llm freeze + prompt
python main.py -d taxi --peft_type freeze --use_prompt
python main.py -d taobao --peft_type freeze --use_prompt
python main.py -d amazon --peft_type freeze --use_prompt
python main.py -d retweet --peft_type freeze --use_prompt


#====================================================================

## Qwen2-1.5B
# lora finetune
python main.py --model_name Qwen2-1.5B -d taxi --peft_type lora --regen_embed
python main.py --model_name Qwen2-1.5B -d taobao --peft_type lora 
python main.py --model_name Qwen2-1.5B -d amazon --peft_type lora
python main.py --model_name Qwen2-1.5B -d retweet --peft_type lora

# llm freeze
python main.py --model_name Qwen2-1.5B -d taxi --peft_type freeze
python main.py --model_name Qwen2-1.5B -d taobao --peft_type freeze
python main.py --model_name Qwen2-1.5B -d amazon --peft_type freeze
python main.py --model_name Qwen2-1.5B -d retweet --peft_type freeze

# --------------------------------------------------------------

# lora finetune + prompt
python main.py --model_name Qwen2-1.5B -d taxi --peft_type lora --use_prompt
python main.py --model_name Qwen2-1.5B -d taobao --peft_type lora --use_prompt
python main.py --model_name Qwen2-1.5B -d amazon --peft_type lora --use_prompt
python main.py --model_name Qwen2-1.5B -d retweet --peft_type lora --use_prompt

# llm freeze + prompt
python main.py --model_name Qwen2-1.5B -d taxi --peft_type freeze --use_prompt
python main.py --model_name Qwen2-1.5B -d taobao --peft_type freeze --use_prompt
python main.py --model_name Qwen2-1.5B -d amazon --peft_type freeze --use_prompt
python main.py --model_name Qwen2-1.5B -d retweet --peft_type freeze --use_prompt