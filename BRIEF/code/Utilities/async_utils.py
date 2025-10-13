import aiohttp
import asyncio
import json
import os
from typing import Union


with open("apikey.json","r") as f:
    API_KEY = json.load(f)["OPENAI_KEY"]
async def chat_local(session, api_key:str, model:str, local_url:str="http://localhost:8000/v1/chat/completions", message:list[dict]=[{"role": "user", "content": "Say this is a test!"}],temperature:float=0,pay_load:dict=None)->Union[dict,None]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    if pay_load:
        payload = {
            **pay_load,
            "model": model,
            "messages": message,
            "temperature": temperature
        }   
    else:
        payload = {
            "model": model,
            "messages": message,
            "temperature": temperature
        }  

    try:
        async with session.post(local_url, headers=headers, json=payload) as response:
            data = await response.json()
    except Exception as E:
        print(f"{E}! Please try again later or check your input!")
        return None

    return data


async def chat(session,api_key:str=API_KEY,model:str="gpt-4o-mini",message:list[dict]=[{"role": "user", "content": "Say this is a test!"}],temperature:float=0.7,use_proxy:bool=True,show_usage:bool=False,pay_load:dict=None)->Union[dict,None]:
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
    
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890',
    }   

    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    if pay_load:
        payload = {
            **pay_load,
            "model": model,
            "messages": message,
            "temperature": temperature
        }   
    else:
        payload = {
            "model": model,
            "messages": message,
            "temperature": temperature
        } 


    try:
        if use_proxy:
            async with session.post(url, headers=headers, json=payload, proxy='http://127.0.0.1:7890') as response:
                data = await response.json()
        else:
            async with session.post(url, headers=headers, json=payload) as response:
                data = await response.json()
    except Exception as E:
        print(f"{E}! Please try again later or check your input!")
        return None
    


    finally:
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        os.environ.pop('all_proxy', None)
    if show_usage:
        print(data['usage']['total_tokens'])
    return data