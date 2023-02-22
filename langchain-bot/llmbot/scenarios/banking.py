from typing import Tuple, Dict
from langchain.llms import BaseLLM
from langchain import PromptTemplate

from llmbot.scenarios.base import FrameScenario, ScenarioAction
from llmbot.utils import parse_json_from_text_multiline, remove_json_from_text_multiline


class SimpleBankingScenario(FrameScenario):
    def __init__(self, llm: BaseLLM, prompt: PromptTemplate):
        super().__init__(llm, prompt)

    def _preprocess(self, kwarg_dict: Dict) -> Dict:
        kwarg_dict.update({
            'balance': '10000',
        })
        return kwarg_dict

    def _postprocess(self, response: str) -> Tuple[str, Dict, ScenarioAction]:
        response = response.strip()
        if 'change_topic' in response.lower() or ('změň' in response.lower() and 'téma' in response.lower()):
            return response, {}, ScenarioAction.CHANGE_TOPIC
        else:
            return response, {}, ScenarioAction.REPLY


class BankingScenarioWJSON(SimpleBankingScenario):
    def __init__(self, llm: BaseLLM, prompt: PromptTemplate):
        super().__init__(llm, prompt)

    def _postprocess(self, response: str) -> Tuple[str, Dict, ScenarioAction]:
        response = response.strip()
        print(response)
        values = parse_json_from_text_multiline(response)
        response = remove_json_from_text_multiline(response)
        if 'recipient' in values and 'amount' in values and \
                len(values['recipient']) > 0 and (len(str(values['amount'])) > 0 and int(values['amount']) > 0):
            return response, values, ScenarioAction.CONFIRM
        elif 'change_topic' in response.lower() or ('změň' in response.lower() and 'téma' in response.lower()):
            return response, values, ScenarioAction.CHANGE_TOPIC
        else:
            return response, values, ScenarioAction.REPLY