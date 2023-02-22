import os
import json

from langchain.llms import OpenAI

from llmbot.classifiers import VectorSimilarityTopicClassiffier
from llmbot.scenarios.multiwoz import MultiwozSingleDomainScenario
from llmbot.scenarios.base import ScenarioAction
from llmbot.prompts.multiwoz import restaurant_prompt, restaurant_prompt_with_db
from llmbot.bots.multiwoz.database import MultiWOZDatabase


class MultiWOZBot:
    def __init__(self,
                 topic_classifier: VectorSimilarityTopicClassiffier,
                 data_dir: str):
        self.topic_classifier = topic_classifier
        self.database = MultiWOZDatabase(os.path.join(data_dir, 'database'))
        self.data_dir = data_dir
        gpt3_llm = OpenAI(model_name="text-davinci-003",
                          temperature=0,
                          top_p=0.9,
                          openai_api_key=os.environ.get('OPENAI_API_KEY', ''))
        self.scenarios = {
            'restaurant': (MultiwozSingleDomainScenario(llm=gpt3_llm, prompt=restaurant_prompt),
                           MultiwozSingleDomainScenario(llm=gpt3_llm, prompt=restaurant_prompt_with_db)),
        }

    def run_with_dataset(self, dataset):
        n = 1
        results = {}
        for dialog in dataset['test']:
            if n > 50:
                break
            if len(dialog['services']) != 1:
                continue
            if dialog['services'][0] not in ['restaurant']:
                continue
            n += 1
            history = []
            topic = None
            domain = dialog['services'][0]
            dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
            results[dialogue_id] = []
            for tn in range(0, len(dialog['turns']['utterance']), 2):
                question = dialog['turns']['utterance'][tn]
                print(question)
                if topic is None:
                    topic = self.topic_classifier.classify(question)
                print("Current topic:", topic)
                topic = "restaurant"
                topic_scenario = self.scenarios[topic]
                answer, values, act = topic_scenario[0](history="\n".join(history),
                                                        question=question)
                db_results = self.database.query(domain=topic,
                                                 constraints=self._get_constraints(values))
                print(values, len(db_results))
                topic_scenario[1].database_count = len(db_results)
                answer, _, act = topic_scenario[1](history="\n".join(history),
                                                        question=question)
                if act == ScenarioAction.CHANGE_TOPIC:
                    topic = self.topic_classifier.classify(question)
                elif act == ScenarioAction.CONFIRM:
                    print(values)
                else:
                    print(answer)
                if act != ScenarioAction.CHANGE_TOPIC:
                    history.append(f"Customer: {question}")
                    history.append(f"Assistant: {answer}")
                results[dialogue_id].append({
                    "response": answer,
                    "state": {
                        "restaurant": values
                        },
                    "active_domains": [topic]
                    })
            with open("results.json", "w") as f:
                json.dump(results, f, indent=4)

    def run(self):
        histories = {
            'restaurant': [],
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
            answer, values, act = topic_scenario[0](history="\n".join(histories[topic]),
                                                    question=question)
            print(values)
            db_results = self.database.query(domain=topic,
                                             constraints=self._get_constraints(values))
            print(values, len(db_results))
            topic_scenario[1].database_count = len(db_results)
            answer, values, act = topic_scenario[1](history="\n".join(histories[topic]),
                                                    question=question)
            if act == ScenarioAction.CHANGE_TOPIC:
                topic = self.topic_classifier.classify(question)
                prompt_user = False
            elif act == ScenarioAction.CONFIRM:
                print(values)
            else:
                print(answer)
            if act != ScenarioAction.CHANGE_TOPIC:
                histories[topic].append(f"Customer: {question}")
                histories[topic].append(f"Assistant: {answer}")

    def _get_constraints(self, values):
        constraints = {}
        for key, value in values.items():
            if value is None or len(value) == 0:
                continue
            if key == 'food':
                constraints['food'] = value
            elif key in ['pricerange', 'price']:
                constraints['pricerange'] = value
            elif key == 'area':
                if 'centre' in value or 'downtown' in value:
                    value = 'centre'
                constraints['area'] = value
            elif key == 'name':
                constraints['name'] = value
        return constraints
