import json
import random
from copy import deepcopy
from typing import Dict, Any
from nltk.tokenize import word_tokenize

from langchain.vectorstores import VectorStore

from definitions import SLOT_MAPPINGS

def serialize_state(state: Dict[str, Dict[str, str]]) -> str:
    serialized = ""
    for domain, ds in state.items():
        slot_mapping_defintion = SLOT_MAPPINGS[domain]
        for n, slot in enumerate(slot_mapping_defintion):
            slot = slot[1]
            if slot in ds:
                serialized += f"{n}: {ds[slot]}"
            else:
                serialized += f"{n}: -"
            if n < len(slot_mapping_defintion) - 1:
                serialized += ", "
    return serialized


def parse_state(state: str, default_domain: str) -> Dict[str, str]:
    slot_mapping_defintion = SLOT_MAPPINGS[default_domain]
    state = state.split(", ")
    parsed = {}
    for n, v in enumerate(state):
        try:
            slot, value = v.split(":")
            slot = int(slot)
            value = value.strip()
        except:
            slot = n
            value = v.strip()
        if slot >= len(slot_mapping_defintion):
            continue
        if value != "-":
            parsed[slot_mapping_defintion[slot][1]] = value
    return {default_domain: parsed}


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
        example_domain = example['domain']
        for key, val in example.items():
            if key == 'state':
                example[key] = serialize_state({example_domain: val})
            elif isinstance(val, dict):
                example[key] = json.dumps(val).replace("{", '<').replace("}", '>')
            else:
                example[key] = str(val)
        return example


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
