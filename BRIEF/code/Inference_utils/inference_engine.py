import subprocess
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from typing import Optional,Callable
import logging
from Utilities.global_utils import confirm_action
original_directory = os.getcwd()
log_filename=os.path.join(original_directory, 'logfile.log')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def translate_path_to_origin(path:str):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(original_directory,path)

def summary_engine(summary_model:str,inpath:str,outpath:str,cuda_setting:str="0,1,2,3",per_device_batchsize:str="16",timeout_limit:str="3600",max_new_tokens:str="128",max_source_length:str="1024",source_prefix:str="summarize: ",main_process_port:str="29500",modify_command:Callable[[list[str]],list[str]]=None,profile=False,cache_dir:str=None,num_beams:str="4"
)->str:
    """
    Only accept list of dicts at least with field: "text", "question", "question_id" and "summary". Inference will be applied to each dict. You should use `formulate_data.py` first if directly use TQA dataset.
    
    Will return a list of dicts with field: `reply`(This is actually the summary by our model), `summary`(This is the ground truth if exist), `text`(This is the retrieved top 5 passages),`question_id`, `question` and `Idx`(This idx isn't qid). This function will also create a `./dir_name.txt` which records the directory of generated folder for inference results and also return it.

    Example usage:
        ```
        dir_name = summary_engine(summary_model="google-t5/t5-large",inpath="./Downstream_Dev.json",outpath="./inference-vanilla",main_process_port=29506,cuda_setting="6,7")
        ```
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)

    logger.info("You are now getting summary for 5 100-word wikipeida passage")
    logger.info(f"The setting you are using is:summary_model:{summary_model},inpath:{inpath},outpath:{outpath},cuda_setting:{cuda_setting},per_device_batchsize:{per_device_batchsize},timeout_limit:{timeout_limit},max_new_tokens:{max_new_tokens},max_source_length:{max_source_length},source_prefix:{source_prefix},main_process_port:{main_process_port}")

    inpath = translate_path_to_origin(inpath)
    outpath = translate_path_to_origin(outpath)
    dir_name = translate_path_to_origin("dir_name.txt")
    command = [
        "accelerate", "launch", "--main_process_port", main_process_port, "accelerate_inference.py", 
        "--model_name",summary_model,
        "--dataset_path", inpath,
        "--output_file_path",outpath,
        "--per_device_batchsize",per_device_batchsize,
        "--timeout_limit",timeout_limit,
        "--max_new_tokens",max_new_tokens,
        "--max_source_length",max_source_length,
        "--source_prefix",source_prefix,
        "--dir_name",dir_name,
        "--num_beams",num_beams
        ]
    if profile == True:
        command[4] = "accelerate_inference_profile.py"

    if cache_dir != None:
        command += ["--use_cache",cache_dir]

        
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_setting
    subprocess.run(command, env=env)
    with open(dir_name,"r") as f:
        dir_name = f.read()
    os.chdir(original_directory)
    return dir_name.strip()

def reader_engine(reader_model:str,inference_type:str,outpath:str,inpath:Optional[str]=None,cuda_setting:str="0,1,2,3",main_process_port:str="29500",instruct:str="True",profile=False,downstream_dataset:str="tqa",use_cot_short=False,use_cot_long=False,gpu_util:str="0.9",modify_command:Callable[[list[str]],list[str]]=None):
    # TODO Add Instruct related logic to the reader
    """
    Only take in list of dicts with field: `reply`(This is actually the summary), `summary`(This is the ground truth if exist), `text`(This is the retrieved top 5 passages),`question_id`, `question` and `Idx`(This idx isn't qid). You can optionally choose to omit some field for certain inference type, but we recommend that all is prepared. You should use summary_engine to generate one.
    
    Note:
        Choose from `ours`, `vanilla`, `all_passage`, `proposition` and `none`
    
    Example Usage:
        ```
        reader_engine(reader_model="flan-ul2",inference_type="vanilla",outpath="0710_downstream_reply_vanilla_wiki_dev.json",cuda_setting="6,7",main_process_port="29507")
        reader_engine(reader_model="llama3",inference_type="none",outpath="0705_llama_downstream_reply_none_logit_v3.json",inpath="inference--tmp-summarization-v3-logit-google-t5-t5-large-2024-07-04_00-30-15-2024-07-04_03-28-43/reply.json")

        ```
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    if inpath is None:
        logger.warning("You are not using anything as input. Default to read from `./dir_name.txt`. You can ignore this if this is desired behaviour.")
        inpath = translate_path_to_origin("./dir_name.txt")
        with open(inpath,"r") as f:
            inpath = f.read().strip() 
            inpath = os.path.join(inpath,"reply.json")
    else:
        inpath = translate_path_to_origin(inpath)

    outpath = translate_path_to_origin(outpath)
    logger.info(f"The setting you are using is:reader_model:{reader_model},inpath:{inpath},outpath:{outpath},cuda_setting:{cuda_setting},inference_type:{inference_type},main_process_port:{main_process_port},instruct:{instruct}")
    if "llama" in reader_model.lower():
        logger.info("You are trying to inference with vLLM on Llama3. Make sure you are in the correct environment.")
        command = [
            "python", "vllm_llama3_inference.py",
            "--inference_type",inference_type,
            "--proposition_name_or_path", inpath,
            "--output_path",outpath,
            "--downstream_dataset",downstream_dataset,
            ]
        if instruct == "True":
            command.append("--instruct")
        if use_cot_short == True:
            command.append("--use_cot_short")
        if use_cot_long == True:
            command.append("--use_cot_long")
        if gpu_util != "0.9":
            command.append("--gpu_util")
            command.append(gpu_util)
    elif "prolong" in reader_model.lower():
        logger.info("You are trying to inference with vLLM on Llama3. Make sure you are in the correct environment.")
        command = [
            "python", "vllm_prolong_inference.py",
            "--inference_type",inference_type,
            "--proposition_name_or_path", inpath,
            "--output_path",outpath,
            "--downstream_dataset",downstream_dataset
            ]
        command.append("--instruct")
    elif "film" in reader_model.lower():
        logger.info("You are trying to inference with vLLM on Llama3. Make sure you are in the correct environment.")
        command = [
            "python", "vllm_film_inference.py",
            "--inference_type",inference_type,
            "--proposition_name_or_path", inpath,
            "--output_path",outpath,
            "--downstream_dataset",downstream_dataset
            ]
        command.append("--instruct")
    elif "flan" in reader_model.lower():
        command = [
            "accelerate", "launch", "--main_process_port", main_process_port, "accelerate_flan_ul2_inference.py", 
            "--inference_type",inference_type,
            "--proposition_name_or_path", inpath,
            "--output_path",outpath,
            "--downstream_dataset",downstream_dataset
            ]
        if profile == True:
            command[4] = "accelerate_flan_ul2_inference_profile.py"
    elif "phi" in reader_model.lower():
        command = [
            "python", "vllm_phi_inference.py",
            "--inference_type",inference_type,
            "--proposition_name_or_path", inpath,
            "--output_path",outpath,
            "--downstream_dataset",downstream_dataset
            ]
        if instruct == "True":
            command.append("--instruct")
    # elif "phi" in reader_model.lower():
    #     command = [
    #         "accelerate", "launch", "--main_process_port", main_process_port, "acclerate_phi_3_inference.py", 
    #         "--inference_type",inference_type,
    #         "--proposition_name_or_path", inpath,
    #         "--output_path",outpath,
    #         "--downstream_dataset",downstream_dataset
    #         ]
    else:
        raise NotImplementedError    
    if modify_command:
        command = modify_command(command)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_setting
    subprocess.run(command, env=env)
    os.chdir(original_directory)

def reply_engine(reply_model:str,inpath:str,outpath:str,cuda_setting:str="0,1,2,3",main_process_port:str="29500",batch_size:str="4",profile=False,instruct:str="True",modify_command:Callable[[list[str]],list[str]]=None):
    """
    This function is used to do the labelling of data with logit.

    Only accept list of dicts at least with field: `QuestionId`, `Question` and `Proposition`. Inference will be applied to each dict.
    
    Will return a list of dicts with field: `QuestionId`, `Question`, `Proposition`, `Proposition_Reply`, `Log_Probabilities` and `Sequence_Probability`.
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    inpath = translate_path_to_origin(inpath)
    outpath = translate_path_to_origin(outpath)
    logger.info(f"You are using instruct = {instruct} setting.")
    if "flan" in reply_model:
        if cuda_setting == "0,1,2,3":
            command = [
                "accelerate", "launch", "--main_process_port", main_process_port,"flan_ul2_inference_logit.py",
                "--inpath", inpath,"--outpath",outpath,"--batch_size",batch_size
                ]
        else:
            import fileinput
            for line in fileinput.input("~/.cache/huggingface/accelerate/default_config_2.yaml", inplace=True):
                if "gpu_ids" in line:
                    line = f"gpu_ids: {cuda_setting}\n"
                sys.stdout.write(line)
            command = [
                "accelerate", "launch", "--config_file", "~/.cache/huggingface/accelerate/default_config_2.yaml","--main_process_port", main_process_port,"flan_ul2_inference_logit.py",
                "--inpath", inpath,"--outpath",outpath,"--batch_size",batch_size
                ]

        
        
    # TODO Finish vllm logic
    elif "prompt_llama" in reply_model:
        # raise NotImplementedError
        # Prompt Llama3 means that we are calculating the likelihood of the first choice
        # correct answer in the prompt.
        command = [
            "python", "vllm_llama3_reply_proposition_prompt.py" , 
            "--parallel_size", batch_size,
            "--proposition_name_or_path",inpath,
            "--output_path",outpath
            ]
        if instruct == "True":
            command.append("--instruct")
    elif "llama" in reply_model:
        # raise NotImplementedError
        command = [
            "python", "vllm_llama3_reply_proposition.py" , 
            "--parallel_size", batch_size,
            "--proposition_name_or_path",inpath,
            "--output_path",outpath
            ]
        if instruct == "True":
            command.append("--instruct")
    if modify_command:
        command = modify_command(command)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_setting
    # print(type( batch_size),type(inpath),type(outpath))
    subprocess.run(command, env=env)

    os.chdir(original_directory)

def embed_engine(inpath:str,outpath:str,top_k:str="5",wiki_split:str="psgs_w100.tsv",cuda_setting:str="0,1,2,3",modify_command:Callable[[list[str]],list[str]]=None):
    """
    Must pass in a list of dicts with at least field `Question`, `EntityPages`, `QuestionId`. It will add field `Top_Passages` to each dict. 
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    inpath = translate_path_to_origin(inpath)
    outpath = translate_path_to_origin(outpath)

    command = [
        "python", "contriever_embed_gpu.py",
        "--inpath", inpath,
        "--outpath",outpath,
        "--top_k",f"{top_k}",
        "--wiki_split",wiki_split
        ]
    if modify_command:
        command = modify_command(command)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_setting
    subprocess.run(command, env=env)

    os.chdir(original_directory)

def propositionizer(inpath:str,outpath:str,batch_size:str="256",cuda_setting:str="0,1,2,3",main_process_port:str="29500",modify_command:Callable[[list[str]],list[str]]=None):
    """
    Must pass in document with at least field `Title`, `Proposition` and `Idx`.

    Will return with additional field `Propositions`. 
    
    Note: It can be very often that the json parser fails to parser the output of the propositionizer, in which cases you need to manually address them.

    Example Usage:
        `propositionizer(inpath="./llama_propositions_unseen.json",outpath="./llama_propositions_unseen_decomposed.json",cuda_setting="0,1,2,3")`
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    inpath = translate_path_to_origin(inpath)
    outpath = translate_path_to_origin(outpath)

    command = [
        "accelerate","launch","--main_process_port", main_process_port, "accelerate_propositionize.py", 
        "--inpath", inpath,
        "--outpath",outpath,
        "--batch_size",batch_size
        ]
    if modify_command:
        command = modify_command(command)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_setting
    subprocess.run(command, env=env)

    os.chdir(original_directory)



def set_up_vllm_server(model:str,api_key:str="token-123123",port:str="8000",model_parallel:str="4",cuda_setting="0,1,2,3"):
    """set_up_vllm_server will host a vLLM server with OpenAI compatible api locally.

    Args:
        - model: a str point to the model to host
        - api_key: a str that the HTTP client need to provide
        - model_paralle: a str shows the number of GPUs to use
        - port: a str shows which port the vLLM is served
        - cuda_setting: a str shows which GPUs are available

    Notes:
        - Use it with any Http server. See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html.
    
    Return
        - a `Popen` object that should be dealt with carefully. 
        - process.terminate()
        - try:
            process.wait(timeout=5)  
          except subprocess.TimeoutExpired:
            print("Process did not terminate in time, killing it.")
            process.kill()
            process.wait()  
    """
    command = [
        "python","-m","vllm.entrypoints.openai.api_server","--model",model, 
        "--port", port,
        "--api-key",api_key,
        "--tensor-parallel-size",model_parallel
        ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_setting
    process = subprocess.Popen(command, env=env)
    return process

def stop_vllm_server(model:str=None):
    if model:
        ps_command = f"ps aux | grep 'vllm.entrypoints.openai.api_server --model {model}' | grep -v grep"
    else:
        ps_command = "ps aux | grep 'vllm.entrypoints.openai.api_server' | grep -v grep"


    # Run the command
    process = subprocess.Popen(ps_command, shell=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()

    # Parse the output to get the PID
    lines = output.decode().strip().split('\n')
    if not model:
        print(lines)
        if confirm_action("You are not specifying a certain model. Do you want to continue?(Y/N)"):
            pass
        else:
            return 0
    if lines and lines[0]:
        pid = lines[0].split()[1]
        print(f"Found process with PID: {pid}. Killing it...")
        # Kill the process
        kill_command = f"kill -9 {pid}"
        subprocess.run(kill_command, shell=True)
        print("Process killed.")
    else:
        print("No process found matching the pattern.")

    ps_command = "ps aux | grep 'multidoc_vllm/bin/python -c from multiprocessing.spawn' | grep -v grep"
    # Run the command
    process = subprocess.Popen(ps_command, shell=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    # Parse the output to get the PID
    lines = output.decode().strip().split('\n')
    if lines:
        for line in lines:
            if line:
                pid = line.split()[1]
                print(f"Found process with PID: {pid}. Killing it...")
                # Kill the process
                kill_command = f"kill -9 {pid}"
                subprocess.run(kill_command, shell=True)
                print("Process killed.")
            else:
                print("No process found matching the pattern.")





