import argparse
import pickle
import os

from langchain import OpenAI
import openai

from classifiers import VectorSimilarityTopicClassiffier
from llmbot.scenarios.base import ScenarioAction
from llmbot.scenarios.banking import BankingScenarioWJSON, SimpleBankingScenario
from llmbot.scenarios.multiwoz import MultiwozSingleDomainScenario
from llmbot.prompts.banking import pay_money_prompt, balance_prompt, want_buy_prompt
from llmbot.prompts.multiwoz import restaurant_prompt

openai.api_key = os.environ.get('OPENAI_API_KEY', '')


class BankingBot:
    def __init__(self, topic_classifier: VectorSimilarityTopicClassiffier):
        self.topic_classifier = topic_classifier
        gpt3_llm = OpenAI(model_name="text-davinci-003",
                          temperature=0.5,
                          openai_api_key=os.environ.get('OPENAI_API_KEY', ''))
        self.scenarios = {
            'pay_money': BankingScenarioWJSON(llm=gpt3_llm, prompt=pay_money_prompt),
            'balance': SimpleBankingScenario(llm=gpt3_llm, prompt=balance_prompt),
            'want_buy': SimpleBankingScenario(llm=gpt3_llm, prompt=want_buy_prompt),
        }

    def run(self):
        context = {
            'pay_money': [],
            'balance': [],
            'want_buy': [],
        }
        topic = None
        question, answer = "", ""
        prompt_user = True
        while True:
            if prompt_user:
                question = input(">").strip()
            prompt_user = True
            if topic is None:
                topic = self.topic_classifier.classify(question)
            print("Current topic:", topic)
            topic_scenario = self.scenarios[topic]
            answer, values, act = topic_scenario(history="\n".join(context[topic]),
                                                 question=question)
            if act == ScenarioAction.CHANGE_TOPIC:
                topic = self.topic_classifier.classify(question)
                prompt_user = False
            elif act == ScenarioAction.CONFIRM:
                print(values)
            else:
                print(answer)
            if act != ScenarioAction.CHANGE_TOPIC:
                context[topic].append(f"Zákazník: {question}")
                context[topic].append(f"Asistent: {answer}")
