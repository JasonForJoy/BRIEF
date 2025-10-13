import subprocess
import os
import json

from typing import Union

with open("apikey.json","r") as f:
    API_KEY = json.load(f)["OPENAI_KEY"]



def chat_local(api_key:str,model:str,local_url:str="http://localhost:8000/v1/chat/completions",message:list[dict]=[{"role": "user", "content": "Say this is a test!"}])->Union[dict,None]:
    """chat_local is the local version of OpenAI's chat completion api.

    Args:
        - api_key: a str represents your local apikey
        - model: a str specifies the model you wish to use.
        - message: a list of dict containing the history of conversation

    Note:
        Use this `data["choices"][0]["message"]["content"]` to get the result.    

    Return:
        a dict that looks like:
        ```
        {
            'id': 'chatcmpl-9nVPOFEIijrP7G6PoniMJDiu7U5Fl', 
            'object': 'chat.completion', 
            'created': 1721585834, 
            'model': 'gpt-4o-mini-2024-07-180/bin/', 
            'choices': [{
                'index': 0, 
                'message': {'role': 'assistant', 'content': 'This is a test! How can I assist you further?'},
                'logprobs': None, 
                'finish_reason': 'stop'
            }], 
            'usage': {'prompt_tokens': 13, 'completion_tokens': 12, 'total_tokens': 25}, 
            'system_fingerprint': 'fp_661538dc1f'
        }
        ```
    """
    import requests
    # Set environment variables
    url = local_url
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    payload = {
        "model": model,
        "messages": message
    }   


    response = requests.post(url, headers=headers, json=payload)
    # Print the output
    try:
        data = json.loads(response.text)
    except Exception as E:
        print("{E}! Please try again later or check your input!")
        return None
    return data

def chat(api_key:str=API_KEY,model:str="gpt-4o-mini",message:list[dict]=[{"role": "user", "content": "Say this is a test!"}],temperature:float=0.7,use_proxy:bool=True,pay_load:dict=None)->Union[dict,None]:
    """chat uses OpenAI's api service to provide a chat interface. This function uses proxy automatically to avoid any potential issue.

    Args:
        - api_key: a str represents you OpenAI apikey
        - model: a str specifies the model you wish to use. You can browse https://platform.openai.com/docs/api-reference/
        - message: a list of dict containing the history of conversation
        - temperature: a float representing the temperature of generation
        - use_proxy: a boolean represent whether to use proxy 127.0.0.1:7890 for http/https

    Note:
        Use this `data["choices"][0]["message"]["content"]` to get the result.    

    Return:
        a dict that looks like:
        ```
        {
            'id': 'chatcmpl-9nVPOFEIijrP7G6PoniMJDiu7U5Fl', 
            'object': 'chat.completion', 
            'created': 1721585834, 
            'model': 'gpt-4o-mini-2024-07-180/bin/', 
            'choices': [{
                'index': 0, 
                'message': {'role': 'assistant', 'content': 'This is a test! How can I assist you further?'},
                'logprobs': None, 
                'finish_reason': 'stop'
            }], 
            'usage': {'prompt_tokens': 13, 'completion_tokens': 12, 'total_tokens': 25}, 
            'system_fingerprint': 'fp_661538dc1f'
        }
        ```
    """
    import requests
    proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
    }   
    # Set environment variables
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    payload = {
        **pay_load,
        "model": model,
        "messages": message,
        "temperature": temperature
    }   
    # print(payload)
    # Execute the curl command and capture the output
    if use_proxy:
        response = requests.post(url, headers=headers, json=payload, proxies=proxies)
    else:
        response = requests.post(url, headers=headers, json=payload)
    # Print the output
    try:
        data = json.loads(response.text)
    except Exception as E:
        print("{E}! Please try again later or check your input!")
        return None
    finally:
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        os.environ.pop('all_proxy', None)
    return data


def chat_sys(api_key:str=API_KEY,model:str="gpt-4o-mini",message:list[dict]=[{"role": "user", "content": "Say this is a test!"}],temperature:float=0.7)->Union[dict,None]:
    """chat uses OpenAI's api service to provide a chat interface. This function uses proxy automatically to avoid any potential issue.

    Args:
        api_key: a str represents you OpenAI apikey
        model: a str specifies the model you wish to use. You can browse https://platform.openai.com/docs/api-reference/
        message: a list of dict containing the history of conversation
        temperature: a float representing the temperature of generation

    Note:
        Use this `data["choices"][0]["message"]["content"]` to get the result.    

    Return:
        a dict that looks like:
        ```
        {
            'id': 'chatcmpl-9nVPOFEIijrP7G6PoniMJDiu7U5Fl', 
            'object': 'chat.completion', 
            'created': 1721585834, 
            'model': 'gpt-4o-mini-2024-07-180/bin/', 
            'choices': [{
                'index': 0, 
                'message': {'role': 'assistant', 'content': 'This is a test! How can I assist you further?'},
                'logprobs': None, 
                'finish_reason': 'stop'
            }], 
            'usage': {'prompt_tokens': 13, 'completion_tokens': 12, 'total_tokens': 25}, 
            'system_fingerprint': 'fp_661538dc1f'
        }
        ```
    """
    # Set environment variables
    os.environ['no_proxy'] = 'localhost,127.0.0.1'
    os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'

    # Define the curl command
    curl_command = [
        'curl', 'https://api.openai.com/v1/chat/completions',
        '-H', 'Content-Type: application/json',
        '-H', f'Authorization: Bearer {api_key}',
        '-d', json.dumps({
            "model": model,
            "messages": message,
            "temperature": temperature
        })
    ]

    # Execute the curl command and capture the output
    result = subprocess.run(curl_command, capture_output=True, text=True)

    # Print the output
    try:
        data = json.loads(result.stdout)
    except Exception as E:
        print("{E}! Please try again later or check your input!")
        return None
    finally:
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        os.environ.pop('all_proxy', None)
    return data





def embedding(api_key:str=API_KEY,model:str="text-embedding-ada-002",text:Union[str,list[str]]="This is a test!",encoding_format:str="float",use_proxy:bool=True)->Union[dict,None]:
    """chat uses OpenAI's api service to provide a embedding. This function uses proxy automatically to avoid any potential issue.

    Args:
        - api_key: a str represents you OpenAI apikey
        - model: a str specifies the model you wish to use. You can browse https://platform.openai.com/docs/api-reference/
        - text: a str or a list of str containing the text you want to embed
        - encoding_format: a str representing the format to return the embeddings in. Can be either float or base64.
        - use_proxy: a boolean represent whether to use proxy 127.0.0.1:7890 for http/https

    Note:
        Use this `data["data"][0]["embedding"]` to get the result.    

    Return:
        a dict that looks like:
        ```
        {
            "object": "list",
            "data": [
                {
                "object": "embedding",
                "embedding": [
                    0.0023064255,
                    -0.009327292,
                    .... (1536 floats total for ada-002)
                    -0.0028842222,
                ],
                "index": 0
                }
            ],
            "model": "text-embedding-ada-002",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
            }
        }
        ```
    """
    import requests
    proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
    }   
    # Set environment variables
    url = 'https://api.openai.com/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    payload = {
        "model": model,
        "input": text,
        "encoding_format": encoding_format
    }   

    # Execute the curl command and capture the output
    if use_proxy:
        response = requests.post(url, headers=headers, json=payload, proxies=proxies)
    else:
        response = requests.post(url, headers=headers, json=payload)
    # Print the output
    try:
        data = json.loads(response.text)
    except Exception as E:
        print("{E}! Please try again later or check your input!")
        return None
    finally:
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        os.environ.pop('all_proxy', None)
    return data










def propositionize(passage:str,max_iter:int = 5,model:str="gpt-4o-mini")->Union[list[dict],None]:
    """ propositionize make a long paragraph into a list of propositions.

    Args:
        - passage: a string that needs to be propositionized
        - max_iter: a int that means how many times allowed for trial-and-error
        - model: a string represents which model to use

    Return:
        a list of dict if all successful
    """
    prompt = f"Input: {passage}\nOutput: "
    prompt = instructions + "\n\n" + one_shot + "\n\n" + prompt
    message = [{"role":"user","content":prompt}]

    def get_wrapped(s:str)->str:
        s = s.removeprefix("```json").removesuffix("```")
        s = s.removeprefix("```").removesuffix("```")
        return s.strip()
    
    for i in range(max_iter):
        data = chat(api_key=API_KEY,model=model,temperature=0,message=message)
        out = data["choices"][0]["message"]["content"]
        out = get_wrapped(out)
        try:
            d = json.loads(out)
            assert len(d) != 0 
            return d
        except Exception as E:
            print(E)
            print(out)
            continue
    print(f"Don't finish in {max_iter} rounds.")
    return None




instructions = """
Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.
4. Present the results as a list of strings, formatted in JSON.
"""
one_shot = """
Input: Title: Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content: The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and both occur on grassland and are first seen in the spring. In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America where it evolved into the Easter Bunny."
Output: ["The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter", "Hares were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America."]
"""
