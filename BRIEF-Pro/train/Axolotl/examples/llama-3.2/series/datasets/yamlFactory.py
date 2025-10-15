"""This file generates the yaml config file for Axolotl framework"""
# Training data paths
# prefix = "../../MuSiQue/long_context/eval/"
prefix = "../MultiHopRAG/eval/"
trainPaths = {
    "prop-eps0.005-supporting": f"{prefix}train_chat_alpaca_prop_eps0.005_supp.jsonl",
    "large_strict": f"{prefix}train_chat_alpaca_prop_force_large_supp.jsonl",
    "small_strict": f"{prefix}train_chat_alpaca_prop_force_small_supp.jsonl",
    "oracle-supporting": f"{prefix}train_chat_alpaca_supporting_prop.jsonl",
    "small_cont": f"{prefix}train_chat_alpaca_prop_cont_small_supp.jsonl",
    "cont": f"{prefix}train_chat_alpaca_prop_cont_supp.jsonl",
    "sent_cont": f"{prefix}train_chat_alpaca_sent_force_cont.jsonl",
    "sent": f"{prefix}train_chat_alpaca_sent_force.jsonl",
    "multihopRAG": f"{prefix}train_chat_alpaca_multihoprag.jsonl",
    "simplemix": f"{prefix}train_chat_alpaca_multihoprag_sent_force_cont.jsonl",
}

evalPaths = {   
    "prop-eps0.005-supporting": f"{prefix}eval_chat_alpaca_prop_eps0.005_supp.jsonl",
    "large_strict": f"{prefix}eval_chat_alpaca_prop_force_large_supp.jsonl",
    "small_strict": f"{prefix}eval_chat_alpaca_prop_force_small_supp.jsonl",
    "oracle-supporting": f"{prefix}eval_chat_alpaca_supporting_prop.jsonl",
    "small_cont": f"{prefix}eval_chat_alpaca_prop_cont_small_supp.jsonl",
    "cont": f"{prefix}eval_chat_alpaca_prop_cont_supp.jsonl",
    "sent_cont": f"{prefix}eval_chat_alpaca_sent_force_cont.jsonl",
    "sent": f"{prefix}eval_chat_alpaca_sent_force.jsonl",
    "multihopRAG": f"{prefix}eval_chat_alpaca_multihoprag.jsonl",
    "simplemix": f"{prefix}eval_chat_alpaca_multihoprag_sent_force_cont.jsonl",
}

settings = [
    "prop-eps0.005-supporting", #0
    "oracle-supporting",        #1
    "large_strict",             #2
    "small_strict",             #3
    "cont",                     #4
    "small_cont",               #5
    "sent_cont",                #6
    "sent",                     #7
    "multihopRAG",              #8
    "simplemix",                 #9
    ]
setting = settings[9]

# Training parameters
# default
# sequenceLen = 4096
# gradAccum = 4
# microBatchSize = 4
# epoch = 3
# lr = 0.0002

evalMaxNewTokens = 512
sequenceLen = 16384
gradAccum = 32
microBatchSize = 1
epoch = 3
lr = 0.0001

inputTrainPath = trainPaths[setting]
inputEvalPath = evalPaths[setting]
setting = setting.replace("-","_").replace(".","_")
outputDir = f"./outputs/lora-out-ins-epoch{epoch}-lr{str(lr).replace('.','_')}-{setting}"
wandbName = f"instruct-r32-neft-lr{str(lr).replace('.','_')}-ep{epoch}-{setting}"


# YAML template
yamlStr = """base_model: ../../selfrag_model_cache/Llama-3.2-3B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: false

chat_template: llama3
datasets:
  - path: {inputTrainPath}
    type: chat_template
    field_messages: messages
    message_field_role: role
    message_field_content: content
    roles:
      user:
        - user
      assistant:
        - assistant

dataset_prepared_path: ./last_run_prepared
test_datasets:
  - path: {inputEvalPath}
    ds_type: json
    # You need to specify a split. For "json" datasets the default split is called "train".
    split: train
    type: chat_template
    field_messages: messages
    message_field_role: role
    message_field_content: content
    roles:
      user:
        - user
      assistant:
        - assistant
    data_files:
      - {inputEvalPath}

output_dir: {outputDir}

sequence_len: {sequenceLen}
sample_packing: true
pad_to_sequence_len: true

adapter: lora
lora_model_dir:
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out: false

loraplus_lr_ratio: 16 # loraplus learning rate ratio lr_B / lr_A. Recommended value is 2^4.
loraplus_lr_embedding: 0.000001
#  loraplus learning rate for lora embedding layers. Default value is 1e-6.
lora_modules_to_save:
  - embed_tokens
  - lm_head

wandb_mode: 
wandb_project: brief-multihoprag
wandb_entity:
wandb_watch: all
wandb_name: {wandbName}
wandb_log_model: checkpoint

gradient_accumulation_steps: {gradAccum}
micro_batch_size: {microBatchSize}
num_epochs: {epoch}
optimizer: adamw_torch
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 0.00000001
lr_scheduler: cosine
learning_rate: {lr}
cosine_min_lr_ratio: 0.1


neftune_noise_alpha: 5 # default value in the NEFT paper

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
# early_stopping_patience: 2
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true
s2_attention:

warmup_steps: 10
evals_per_epoch: 2
# mutex with eval_steps
# eval_steps: 319
# These are not properly maintained
# !eval_table_size: 2
eval_batch_size: 4
eval_max_new_tokens: {evalMaxNewTokens}
# However, the false must be set in order to do causal lm eval
eval_sample_packing: false # false will lead to drop_last, which will cause potential error
eval_causal_lm_metrics: ['chrf','sacrebleu']
do_causal_lm_eval: true

saves_per_epoch: 1 # mutex with save_steps
# save_steps: 319

# deepspeed: deepspeed_configs/zero1.json
ddp_timeout: 720000
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
   pad_token: "<|end_of_text|>"
   # According to https://github.com/axolotl-ai-cloud/axolotl/discussions/1812
   # Note: <|end_of_text|> for the base model; and <|eot_id|> for the chat model
   bos_token: "<|begin_of_text|>"
   eos_token: "<|eot_id|>"


seed: 42
strict: false
debug: true
# trust_remote_code: true # this seems to lead to stupid CUDA memory issue, perhaps some is touching """


yaml = yamlStr.format(inputTrainPath=inputTrainPath, inputEvalPath=inputEvalPath, outputDir=outputDir, sequenceLen=sequenceLen, wandbName=wandbName, gradAccum=gradAccum, microBatchSize=microBatchSize, epoch=epoch, lr=lr, evalMaxNewTokens=evalMaxNewTokens)


with open(f"factory/instruct-lora-3b-ep{epoch}-lr{lr}-{setting}".replace(".","_")+".yml","w") as f:
    f.write(yaml)
