import logging
import os
from collections import namedtuple
import json

def get_logger(name='default_logger', log_file='logs/default.log', level=logging.INFO):
    """Function to get a logger instance.
    
    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file relative to the src directory.
        level (int): Logging level.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, '..', log_file)

    log_dir_path = os.path.dirname(log_dir)
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir)
        file_handler.setLevel(level)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def get_raw_data(data_source:str):
    """Function to get raw training data of trivia_qa.

    Args:
        data_source (str): Path to the `json` file of TriviaQA training data.

    Returns:
        dict[str,TriviaQA]: Dict using `QuestionId` as key and `TriviaQA` as value.

    Note:
        `TriviaQA = namedtuple("TriviaQA",["Answer","EntityPages","Question"])`
    """
    TriviaQA = namedtuple("TriviaQA",["Answer","EntityPages","Question"])
    with open(data_source,"r") as f:
        wiki = json.load(f)
    wiki = wiki['Data']
    wiki:dict[str,TriviaQA] = {w["QuestionId"]:TriviaQA(w["Answer"],w["EntityPages"],w["Question"]) for w in wiki}
    return wiki



def confirm_action(prompt="Are you sure you want to proceed? (Y/N): "):
    """
    # Example usage:
    ```
        if confirm_action("This action may be dangerous. Do you want to continue? (Y/N): "):
            # Proceed with the action
            print("Proceeding with the action...")
        else:
            # Abort the action
            print("Action aborted.")
    ```
    """
    while True:
        response = input(prompt).strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print("Please respond with 'Y' or 'N'.")


