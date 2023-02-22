from llmbot.scenarios.base import ScenarioAction, FrameScenario
from llmbot.utils import parse_json_from_text_multiline, remove_json_from_text_multiline


class MultiwozSingleDomainScenario(FrameScenario):
    def __init__(self, llm, prompt):
        super().__init__(llm, prompt)
        self.database_count = 100

    def _preprocess(self, kwarg_dict):
        kwarg_dict.update({
            'database_count': self.database_count,
        })
        return kwarg_dict

    def _postprocess(self, response):
        response = response.strip()
        values = parse_json_from_text_multiline(response)
        response = remove_json_from_text_multiline(response)
        if 'change_topic' in response.lower() or ('change' in response.lower() and 'topic' in response.lower()):
            return response, {}, ScenarioAction.CHANGE_TOPIC
        else:
            return response, values, ScenarioAction.REPLY