import argparse
import pickle
import glob
import os
from typing import Dict, List

from datasets import load_dataset
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.docstore.document import Document


def delexicalize(utterance: str, frames):
    for s_idx in range(len(frames['slots'][0]['slot']) - 1, -1, -1):
        name = frames['slots'][0]['slot'][s_idx]
        placeholder = f'[{name}]'
        utterance = utterance[:frames['slots'][0]['start'][s_idx]] + placeholder + utterance[frames['slots'][0]['exclusive_end'][s_idx]:]
    return utterance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_faiss_db')
    parser.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2', help='Embedding model name; text-embedding-ada-002 for OpenAI, sentence-transformers/all-mpnet-base-v2 for HuggingFace')
    parser.add_argument('--context_size', type=int, default=3)
    parser.add_argument('--embeddings', default='huggingface', help='huggingface or openai')
    args = parser.parse_args()

    docs = []
    if args.embeddings == 'huggingface':
        embeddings = HuggingFaceEmbeddings(model_name=args.model)
    else:
        embeddings = OpenAIEmbeddings(document_model_name=args.model,
                                      query_model_name=args.model,
                                      openai_api_key=os.environ.get('OPENAI_API_KEY', ''))

    dataset = load_dataset('schema_guided_dstc8')
    n = 1
    for dialog in dataset['train'].shuffle():
        if n > 500:
            break
        if len(dialog['services']) != 1:
            continue
        n += 1
        domain_gt = dialog['services'][0].split('_')[0].lower()
        last_state = {}
        for tn in range(0, len(dialog['turns']['utterance']), 2):
            context = [f"Customer: {t}" if n % 2 == 0 else f"Assistant: {t}"
             for n, t in enumerate(dialog['turns']['utterance'][:tn+1])]
            state = dialog['turns']['frames'][tn]['state']
            if len(state) == 0:
                state = {}
            else:
                state = state[0]['slot_values']
                state = {k: v[0] for k, v in zip(state['slot_name'], state['slot_value_list']) }
            new_state = {domain_gt: {}}
            for sl, val in state.items():
                new_state[domain_gt][sl] = val
            state_update = {domain_gt: {}}
            for domain, domain_state in new_state.items():
                for slot, value in domain_state.items():
                    if slot not in last_state.get(domain, {}) or last_state[domain][slot] != value:
                        state_update[domain][slot] = value
            last_state = new_state

            database_results = dialog['turns']['frames'][tn+1]['service_results'][0]
            doc = Document(page_content='\n'.join(context[-args.context_size:]),
                           metadata={'domain': domain_gt,
                                     'state': state_update,
                                     'context': '\n'.join(context),
                                     'response': delexicalize(dialog['turns']['utterance'][tn+1], dialog['turns']['frames'][tn+1]),
                                     'database': {domain_gt: len(database_results['service_results_list'])}})
            docs.append(doc)
    faiss_vs = FAISS.from_documents(documents=docs,
                                    embedding=embeddings)
    with open(args.output_faiss_db, 'wb') as f:
        pickle.dump(faiss_vs, f)
