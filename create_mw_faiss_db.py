import argparse
import pickle
import glob
import os
from typing import Dict, List

from datasets import load_dataset
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.docstore.document import Document

from database import MultiWOZDatabase

def delexicalize(utterance: str, span_info: Dict[str, List[str]]):
    for s_idx in range(len(span_info['act_slot_name']) - 1, -1, -1):
        name = span_info['act_slot_name'][s_idx]
        placeholder = f'[{name}]'
        utterance = utterance[:span_info['span_start'][s_idx]] + placeholder + utterance[span_info['span_end'][s_idx]:]
    return utterance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_faiss_db')
    parser.add_argument('--model', default='text-embedding-ada-002', help='Embedding model name; text-embedding-ada-002 for OpenAI, sentence-tranformers/all-mpnet-base-v2 for HuggingFace')
    parser.add_argument('--database_path', default='multiwoz_database')
    parser.add_argument('--context_size', type=int, default=3)
    parser.add_argument('--embeddings', default='huggingface', help='huggingface or openai')
    args = parser.parse_args()

    docs = []
    if args.embeddings == 'huggingface':
        embeddings = HuggingFaceEmbeddings(repo_id=args.model)
    else:
        embeddings = OpenAIEmbeddings(document_model_name=args.model,
                                      query_model_name=args.model,
                                      openai_api_key=os.environ.get('OPENAI_API_KEY', ''))

    dataset = load_dataset('multi_woz_v22')
    n = 1
    database = MultiWOZDatabase(args.database_path)
    for dialog in dataset['train']:
        if n > 500:
            break
        if len(dialog['services']) != 1:
            continue
        n += 1
        domain = dialog['services'][0]
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
                if domain not in new_state:
                    new_state[domain] = {name: val}
                else:
                    new_state[domain][name] = val
            database_results = {domain: len(database.query(domain, domain_state))
                                for domain, domain_state in new_state.items()}

            doc = Document(page_content='\n'.join(context[-args.context_size:]),
                           metadata={'domain': f'{domain}',
                                     'state': new_state,
                                     'context': '\n'.join(context),
                                     'response': delexicalize(dialog['turns']['utterance'][tn+1],
                                                              dialog['turns']['dialogue_acts'][tn+1]['span_info']),
                                     'database': database_results})
            docs.append(doc)
    faiss_vs = FAISS.from_documents(documents=docs,
                                    embedding=embeddings)
    with open(args.output_faiss_db, 'wb') as f:
        pickle.dump(faiss_vs, f)