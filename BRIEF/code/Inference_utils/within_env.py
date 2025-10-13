import subprocess
from functools import wraps


def run_with_env(func,env:str,cuda_setting:str="0,1,2,3"):
    @wraps(func)
    def env_func(*args,**kwargs):
        print(f"You specified running script with environment: {env}")
        def modify_prompt(input:list[str])->list[str]:
            prefix = ["conda","run","-n",env, f"CUDA_VISIBLE_DEVICES={cuda_setting}"]
            command = prefix + input
            return command
        return func(modify_command=modify_prompt,*args,**kwargs)
    return env_func