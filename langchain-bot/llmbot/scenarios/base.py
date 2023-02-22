from typing import Tuple, Dict
from enum import Enum
from langchain.llms import BaseLLM
from langchain import LLMChain, PromptTemplate


class ScenarioAction(Enum):
    REPLY = "reply"
    CONFIRM = "confirm"
    CHANGE_TOPIC = "change_topic"


class FrameScenario:
    def __init__(self, llm: BaseLLM, prompt: PromptTemplate):
        self.llm = llm
        self.prompt = prompt
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def __call__(self, *args, **kwargs):
        kwargs = self._preprocess(kwargs)
        response = self.chain.run(**kwargs)
        return self._postprocess(response)

    def _preprocess(self, kwarg_dict: Dict) -> Dict:
        return kwarg_dict

    def _postprocess(self, response: str) -> Tuple[str, Dict, ScenarioAction]:
        return response.strip(), {}, ScenarioAction.REPLY
