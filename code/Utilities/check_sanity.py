from openai_utils import chat
import json
import argparse

with open("apikey.json","r") as f:
    API_KEY = json.load(f)["OPENAI_KEY"]

def arg_parser():
    parser = argparse.ArgumentParser(description="Give suggestions for a certain Python file.")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="The path to the Python file you want to check.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="The model you want to use.",
    )
    args = parser.parse_args()
    return args

def check_sanity(file:str,model:str="gpt-4"):
    print(f"Now check sanity for {file}. Using model {model}.")
    with open(file,"r") as f:
        file_content = f.read()
    file_content = "```python\n" + file_content.strip() + "\n```"
    message = [{"role":"user","content":f"{file_content}\nSuppose all imported functions are all well-behaved. Please check if there is any logic error in my code. Especially those rather stupid ones."}]
    data = chat(api_key=API_KEY,model="gpt-4",message=message,temperature=0.5,use_proxy=True)
    print(data["choices"][0]["message"]["content"])



if __name__ == "__main__":
    args = arg_parser()
    check_sanity(file=args.path,model=args.model)