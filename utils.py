import json
import dirtyjson
import random
from copy import deepcopy
from typing import Dict, Any
from nltk.tokenize import word_tokenize

from langchain.vectorstores import VectorStore


def parse_state(state: str, default_domain: str) -> Dict[str, str]:
    state = str(state)
    state = state.replace('<', '{').replace('>', '}')
    try:
        state = dirtyjson.loads(state)
        try:
            for domain, domain_state in state.items():
                for slot, value in domain_state.items():
                    pass

            return state
        except:
            return {default_domain: state}

    except:
        if state.count('{') == 1:
            state = '{ ' + default_domain + ' ' + state
        state_tk = word_tokenize(state)
        # filter only tokens that are alphanumeric or braces
        state_tk = [tk for tk in state_tk if tk.isalpha() or tk in ['{', '}',',']]
        parsed_state = {default_domain: {}}
        level = 0
        current_domain = default_domain 
        idx = 0
        while idx < len(state_tk):
            tk = state_tk[idx]
            if tk == '{':
                # level += 1
                pass
            elif tk == '}':
                # level -= 1
                pass
            # elif level == 1:
            #     current_domain = tk
            #     parsed_state[tk] = {}
            else:
                slot = tk
                value = []
                idx += 1
                if idx >= len(state_tk):
                    break
                while state_tk[idx] not in  [',', '}']:
                    value.append(state_tk[idx])
                    idx += 1
                    if idx >= len(state_tk):
                        break
                parsed_state[current_domain][slot] = ' '.join(value)
            idx += 1
            if idx >= len(state_tk):
                break
        return parsed_state


class ExampleRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, text: str, k: int = 2) -> list[Dict]:
        result = self.vector_store.similarity_search(text, k=k)
        examples = [{'context': doc.metadata['context'],
                     'state': doc.metadata['state'],
                     'response': doc.metadata['response'],
                     'database': doc.metadata['database'],
                     'domain': doc.metadata['domain']}
                     for doc in result]
        return examples
    


class ExampleFormatter:
    def __init__(self, ontology: Dict):
        self.ontology = ontology

    def format(self, 
               examples: list[Dict[str, Any]],
               input_keys: list[str],
               output_keys: list[str],
               corrupt_state: bool = False) -> list[Dict[str, str]]:

        examples = deepcopy(examples)
        if corrupt_state:
            examples = [self._corrupt_example(example) for example in examples]
        for example in examples:
            state_domains = list(example['state'].keys())
            if len(state_domains) > 0:
                example['state'] = example['state'][state_domains[0]] # flatten the state
            else:
                example['state'] = {}
        examples = [self._example_to_str(example) for example in examples]

        def _prepare_example(example: Dict) -> Dict:
            example['input'] = '\n'.join((f"{key}: {example[key]}" for key in input_keys))
            example['output'] = '\n'.join((f"{key}: {example[key]}" for key in output_keys))
            return example
        examples = [_prepare_example(example) for example in examples]

        return examples
    
    def _corrupt_example(self, example: Dict) -> Dict:
        for domain, dbs in example['state'].items():
            for slot, value in dbs.items():
                slot_otgy_name = f"{domain}-{slot}"
                if slot_otgy_name in self.ontology:
                    example['state'][domain][slot] = random.choice(self.ontology[slot_otgy_name])
        return example
    
    def _example_to_str(self, example: Dict) -> Dict:
        for key, val in example.items():
            if isinstance(val, dict):
                example[key] = json.dumps(val).replace("{", '<').replace("}", '>')
            else:
                example[key] = str(val)
        return example


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
