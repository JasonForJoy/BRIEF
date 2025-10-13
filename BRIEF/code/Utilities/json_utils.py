import re
from typing import Union
import json



def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def read_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Strip any leading/trailing whitespace and parse the JSON object
                data.append(json.loads(line.strip()))
        return data





def json_fix_quotes(s:str)->Union[list[str],None]:
    """ find_non_delimiter_quotes assumes the input string is nearly valid JSON string with shape: `["...","...","...",...,"..."]` and the only issue is that there are invalid `"` in side of the `str`.

    Args:
        s: a string that need to be fixed

    Return:
        a list of str; or `None` representing that this method can't correctly fix the JSON string.
    """
    # Find all " characters (escaped or not)
    quotes = [match.start() for match in re.finditer(r'(?<!\\)"', s)]
    # Find all delimiters
    delimiters = [match.span() for match in re.finditer(r'(\[\s*")|("\s*\])|(",\s*")', s)]
    # Filter out the quotes that are part of a delimiter
    non_delimiter_quotes = [q for q in quotes if not any(start <= q < end for start, end in delimiters)]

    non_delimiter_quotes.reverse()

    # Insert a backslash before each non-delimiter quote
    s_list = list(s)
    for i in non_delimiter_quotes:
        s_list.insert(i, '\\')

    correct_s = ''.join(s_list)
    try:
        data = json.loads(correct_s)
        return data
    except:
        print(f"Can't parse: {s[:10]}")
        return None
