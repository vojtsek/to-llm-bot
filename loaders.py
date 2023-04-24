from datasets import load_dataset
from collections import defaultdict
from typing import Dict, List
from database import MultiWOZDatabase


def load_mwoz(database_path, context_size, split='train', total=10, shuffle=True, available_domains=None, only_single_domain=False, restrict_domains=None):
    database = MultiWOZDatabase(database_path)
    dataset = load_dataset('multi_woz_v22')
    if available_domains is not None:
        domain_counts = {d: 0 for d in available_domains}
    else:
        domain_counts = defaultdict(int)
        domain_counts['aux'] = -1
    if shuffle:
        data = dataset[split].shuffle()
    else:
        data = dataset[split]
    n = 1
    slots_per_domain = defaultdict(set)
    domain_counter = defaultdict(int)
    for dialog in data:
        if only_single_domain and len(dialog['services']) != 1:
            continue
        if all((dc >= total for dc in domain_counts.values())) or (available_domains is None and n >= total):
            break
        dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
        if len(dialog['services']) > 0:
            domain_gt = dialog['services'][0]
        else:
            domain_gt = ''
        for dom in dialog['services']:
            domain_counter[dom] += 1
        if restrict_domains is not None and not all((dom in restrict_domains for dom in dialog['services'])):
            continue
        if domain_counts[domain_gt] >= total:
            continue
        domain_counts[domain_gt] += 1
        n + 1
        last_state = {}
        for tn in range(0, len(dialog['turns']['utterance']), 2):
            context = [f"Customer: {t}" if n % 2 == 0 else f"Assistant: {t}"
             for n, t in enumerate(dialog['turns']['utterance'][:tn+1])]
            state = dialog['turns']['frames'][tn]['state']
            if len(state) == 0:
                state = {}
            else:
                state = state[0]['slots_values']
                state = {k: v[0] for k, v in zip(state['slots_values_name'], state['slots_values_list']) }
            new_state = {}
            for sl, val in state.items():
                domain, name = sl.split('-')
                slots_per_domain[domain].add(name)
                if domain not in new_state:
                    new_state[domain] = {name: val}
                else:
                    new_state[domain][name] = val
            state_update = {}
            for domain, domain_state in new_state.items():
                for slot, value in domain_state.items():
                    if slot not in last_state.get(domain, {}) or last_state[domain][slot] != value:
                        if domain not in state_update:
                            state_update[domain] = {}
                        state_update[domain][slot] = value
            last_state = new_state
            database_results = {domain: len(database.query(domain, domain_state))
                                for domain, domain_state in new_state.items()}

            turn = {'page_content': '\n'.join(context[-context_size:]),
                    'question': dialog['turns']['utterance'][tn],
                    'gt_state': last_state,
                    'dialogue_id': dialogue_id,
                    'metadata': {'domain': f'{domain_gt}',
                                 'state': state_update,
                                 'full_state': last_state,
                                 'context': '\n'.join(context[-6:]),
                                 'response': delexicalize_mwoz(dialog['turns']['utterance'][tn+1],
                                                               dialog['turns']['dialogue_acts'][tn+1]['span_info']),
                                 'database': database_results}}
            yield turn
    print(slots_per_domain)
    print(domain_counter)


def load_sgd(context_size, split='train', total=10, shuffle=True, available_domains=None, only_single_domain=False, restrict_domains=None):
    dataset = load_dataset('schema_guided_dstc8')
    if available_domains is not None:
        domain_counts = {d: 0 for d in available_domains}
    else:
        domain_counts = defaultdict(int)
        domain_counts['aux'] = -1
    n = 1
    if shuffle:
        import transformers
        transformers.set_seed(42)
        data = dataset[split].shuffle()
    else:
        data = dataset[split]
    all_domain_slots = {}
    for dialog in data:
        if only_single_domain and len(dialog['services']) != 1:
            continue
        if all((dc >= total for dc in domain_counts.values())) or (available_domains is None and n >= total):
            break
        domain_gt = dialog['services'][0].split('_')[0].lower()
        if available_domains is not None and domain_gt not in available_domains:
            continue
        if domain_counts[domain_gt] >= total:
            continue
        n += 1
        domain_counts[domain_gt] += 1
        if domain_gt not in all_domain_slots:
            all_domain_slots[domain_gt] = set()
        last_state = {}
        for tn in range(0, len(dialog['turns']['utterance']), 2):
            context = [f"Customer: {t}" if n % 2 == 0 else f"Assistant: {t}"
             for n, t in enumerate(dialog['turns']['utterance'][:tn+1])]
            state = dialog['turns']['frames'][tn]['state']
            requested_slots = state[0]['requested_slots']
            if len(state) == 0:
                state = {}
            else:
                state = state[0]['slot_values']
                state = {k: v[0] for k, v in zip(state['slot_name'], state['slot_value_list']) }
            new_state = {domain_gt: {}}
            for sl, val in state.items():
                all_domain_slots[domain_gt].add(sl)
                new_state[domain_gt][sl] = val
            state_update = {}
            for domain, domain_state in new_state.items():
                for slot, value in domain_state.items():
                    if slot not in last_state.get(domain, {}) or last_state[domain][slot] != value:
                        if domain not in state_update:
                            state_update[domain] = {}
                        state_update[domain][slot] = value
            last_state = new_state

            database_results = dialog['turns']['frames'][tn+1]['service_results'][0]
            turn= {'page_content': '\n'.join(context[-context_size:]),
                   'question': dialog['turns']['utterance'][tn],
                   'gt_state': last_state,
                   'dialogue_id': dialog['dialogue_id'],
                   'requested_slots': requested_slots,
                   'metadata': {'domain': domain_gt,
                                'state': state_update,
                                'full_state': last_state,
                                'context': '\n'.join(context),
                                'response': delexicalize_sgd(dialog['turns']['utterance'][tn+1], dialog['turns']['frames'][tn+1]),
                                'database': {domain_gt: len(database_results['service_results_list'])}}}
            yield turn


def delexicalize_mwoz(utterance: str, span_info: Dict[str, List[str]]):
    for s_idx in range(len(span_info['act_slot_name']) - 1, -1, -1):
        name = span_info['act_slot_name'][s_idx]
        placeholder = f'[{name}]'
        utterance = utterance[:span_info['span_start'][s_idx]] + placeholder + utterance[span_info['span_end'][s_idx]:]
    return utterance


def delexicalize_sgd(utterance: str, frames):
    for s_idx in range(len(frames['slots'][0]['slot']) - 1, -1, -1):
        name = frames['slots'][0]['slot'][s_idx]
        placeholder = f'[{name}]'
        utterance = utterance[:frames['slots'][0]['start'][s_idx]] + placeholder + utterance[frames['slots'][0]['exclusive_end'][s_idx]:]
    return utterance


