import json
import re
import dirtyjson
import random
from copy import deepcopy
from typing import Dict, Any
from collections import defaultdict

import numpy
from fuzzywuzzy import fuzz
import evaluate
from nltk.tokenize import word_tokenize
from langchain.vectorstores import VectorStore

from loaders import load_sgd


def parse_state(state: str, default_domain: str = None) -> Dict[str, str]:
    def sanitize(dct):
        for key in dct:
            if isinstance(dct[key], dict):
                dct[key] = sanitize(dct[key])
            elif not isinstance(dct[key], str):
                dct[key] = str(dct[key])
        return dct

    state = str(state)
    slotvals = re.findall("('[a-z]+': ?('(([a-z]| |[A-Z]|:|[0-9])+')|[A-Za-z0-9:]+))", state)
    # slotvals = re.findall("([a-z]+:('(([a-z]| |[A-Z]|:|[0-9])+')|[A-Za-z0-9:]+))", state)
    out_state = {}
    for sv in slotvals:
        sv = sv[0].strip("'\"").split(':')
        out_state[sv[0].strip("'\"")] = ":".join(sv[1:]).strip("'\" ")
    return sanitize(out_state)

    if not state.startswith("{"):
        state = "{" + state
    if not state.endswith("}"):
        state = state + '}'
    state = state.replace('<', '{').replace('>', '}')
    try:
        state = dirtyjson.loads(state)
        try:
            for domain, domain_state in state.items():
                for slot, value in domain_state.items():
                    pass

            return sanitize(state)
        except:
            return {default_domain: sanitize(state)}

    except:
        state = str(state)
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
        return sanitize(parsed_state)


class ExampleRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, text: str, k: int = 2) -> list[Dict]:
        result = self.vector_store.similarity_search(text, k=k)
        examples = [{'context': doc.metadata['context'],
                     'state': doc.metadata['state'],
                     'full_state': doc.metadata['full_state'],
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
               use_json: bool = False,
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
        examples = [self._example_to_str(example, use_json) for example in examples]

        def _prepare_example(example: Dict) -> Dict:
            example['input'] = '\n'.join((f"{key if key != 'full_state' else 'state'}: {example[key]}" for key in input_keys))
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
                else:
                    otgy_key = random.choice(list(self.ontology.keys()))
                    example['state'][domain][slot] = random.choice(self.ontology[otgy_key])
        return example
    
    def _example_to_str(self, example: Dict, use_json=False) -> Dict:
        for key, val in example.items():
            if isinstance(val, dict):
                if use_json:
                    example[key] = json.dumps(val) # .replace("{", '<').replace("}", '>')
                else:
                    example[key] = "-".join((f"{slot}:'{value}'" for slot, value in val.items()))
            else:
                example[key] = str(val)
        return example


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


class SGDEvaluator:
    def __init__(self, split):
        self.data = {}
        self.sacrebleu = evaluate.load('sacrebleu')
        self.bertscore = evaluate.load('bertscore')
        for turn in load_sgd(1, split, total=100000, shuffle=False):
            if turn['dialogue_id'] not in self.data:
                self.data[turn['dialogue_id']] = []
            self.data[turn['dialogue_id']].append({
                'question': turn['question'],
                'state': turn['gt_state'],
                'domain': turn['metadata']['domain'],
                'requested_slots': turn['requested_slots'],
                'response': turn['metadata']['response']
            })

    def get_bleu(self, input_data):
        predictions = []
        references = []
        simple_references = []
        for dialogue_id in input_data:
            for tn, turn in enumerate(input_data[dialogue_id]):
                predictions.append(turn['response'])
                references.append([self.data[dialogue_id][tn]['response']])
                simple_references.append(self.data[dialogue_id][tn]['response'])
        results = self.sacrebleu.compute(predictions=predictions, references=references)
        output = {'bleu': results['score']}
        results_bertscore = self.bertscore.compute(predictions=predictions, references=simple_references, lang='en')
        output['bertscore-f1'] = numpy.mean(results_bertscore['f1'])
        return output

    def get_eval(self, input_data):
        def f1(results):
            epsilon = 0.0000000001
            precision = results['tp'] / (results['tp'] + results['fp'] + epsilon)
            recall = results['tp'] / (results['tp'] + results['fn'] + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)
            return precision, recall, f1

        def extract_placeholders(utt):
            placeholders = re.findall('\[[^ ]*\]', utt)
            placeholders = [p.lower().replace('_', ' ') for p in placeholders]
            placeholders = [p for p in placeholders if all([k not in p for k in ['address', 'phone', 'number', 'postcode']])]
            placeholders = [p.replace('street', '').strip('[]') for p in placeholders]
            return placeholders

        domain_detections = []
        all_turns_scores = []
        successes = []
        turn_successes = []
        slot_results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        total_results = {'tp': 0, 'fp': 0, 'fn': 0}
        for dialog_id in input_data:
            all_provided_gold = set()
            all_provided = set()
            all_informed_gold = set()
            all_informed = set()
            for i, turn in enumerate(input_data[dialog_id]):
                domain_detections.append(turn['domain'] == self.data[dialog_id][i]['domain'])
                response_hyp = turn['response']
                gold_response = self.data[dialog_id][i]['response']
                # gold_requested = set(extract_placeholders(gold_response))
                gold_provided = self.data[dialog_id][i]['requested_slots']
                all_provided_gold.update(self.data[dialog_id][i]['requested_slots'])
                hyp_provided = set(extract_placeholders(response_hyp))
                all_provided.update(hyp_provided)
                gold_state = self.data[dialog_id][i]['state']
                turn_correct = True
                for domain, gold_domain_state in gold_state.items():
                    if domain not in turn["state"]:
                        for slot in gold_domain_state:
                            total_results['fn'] += 1
                            slot_results[slot]['fn'] += 1
                        turn_correct = False
                        continue
                    for slot, value in gold_domain_state.items():
                        all_informed_gold.add(value.lower())
                        if slot not in turn["state"][domain]:
                            turn_correct = False
                            total_results['fn'] += 1
                            slot_results[slot]['fn'] += 1
                        value = value.lower()
                        pred_value = str(turn["state"][domain][slot]).lower() if slot in turn["state"][domain] else ''
                        if fuzz.partial_ratio(value.lower(), pred_value.lower()) <= 0.95:
                            total_results['fn'] += 1
                            slot_results[slot]['fn'] += 1
                            turn_correct = False
                        else:
                            total_results['tp'] += 1
                            slot_results[slot]['tp'] += 1
                for domain, ds in turn["state"].items():
                    for slot, val in ds.items():
                        all_informed.add(str(val).lower())
                        if domain not in gold_state or slot not in gold_state[domain]:
                            total_results['fp'] += 1
                            slot_results[slot]['fp'] += 1
                all_turns_scores.append(int(turn_correct))
                provided_correct = gold_provided == hyp_provided
                turn_successes.append(turn_correct and provided_correct)
            if all_informed_gold.issubset(all_informed) and all_provided_gold.issubset(all_provided):
                successes.append(1)
            else:
                successes.append(0)
        jga = numpy.mean(all_turns_scores)
        prec, recall, micro_f1 = f1(total_results)
        macros = {sl: f1(sl_results)[2] for sl, sl_results in slot_results.items()}
        macro_f1 = numpy.mean([val for val in macros.values()])
        return {'jga': jga, 'micro-F1': micro_f1, 'macro-F1': macro_f1, 'success': numpy.mean(successes), 'turn-success': numpy.mean(turn_successes), 'domain': numpy.mean(domain_detections)}

