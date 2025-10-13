import subprocess
import os
from typing import Optional
import logging
from Inference_utils.wait_until_free import wait_until_free_do
original_directory = os.getcwd()
log_filename=os.path.join(original_directory, 'logfile.log')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

@wait_until_free_do
def train(train_file:str,validation_file:str,output_dir:str,epoch:str,lr:str,warm_up:str,main_process_port:str="29504",seed:str="42",cuda_setting:str="0,1,2,3"):
    command = [
            "accelerate", "launch",
            "--main_process_port", main_process_port,
            # "python",
            "accelerate_sft.py",
            # "--model_name_or_path", "google/flan-t5-large", 
            "--model_name_or_path", "google-t5/t5-large", 
            "--train_file", train_file,
            "--validation_file", validation_file,
            "--source_prefix", "summarize: ",
            "--output_dir", output_dir,
            "--text_column", "text",
            "--summary_column", "summary",
            "--question_column", "question",
            "--with_tracking",
            "--per_device_train_batch_size", "4",
            "--num_train_epochs", epoch,
            "--num_beams", "4",
            "--learning_rate", lr,
            "--num_warmup_steps", warm_up,
            "--lr_scheduler_type", "constant_with_warmup",
            "--seed", seed,
            "--checkpointing_steps", "epoch",
            "--report_to", "wandb"
            ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_setting
    subprocess.run(command, env=env)



train_files=(
    "../Training_Data/HotpotQA-train.json",
    "../Training_Data/MuSiQue-train.json",
    ""
)

validation_files=(
    "../Training_Data/HotpotQA-dev.json",
    "../Training_Data/MuSiQue-dev.json",
    ""
)

output_dirs=(
    "output/HotpotQA",
    "output/MuSiQue",
    ""
)
# 0,1,2,3,...,k-1
k = 2
for i in range(k):
    train_file = train_files[i]
    dev_file = validation_files[i]
    output_dir = output_dirs[i]
    print(f"{train_file},{dev_file},{output_dir}")
    logger.info(f"{train_file},{dev_file},{output_dir}")
    train(train_file=train_file,validation_file=dev_file,output_dir=output_dir,epoch="5",lr="0.00003",warm_up="1000")
