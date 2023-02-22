import argparse
import pickle
import glob
import os

from datasets import load_dataset
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

from database import MultiWOZDatabase


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_faiss_db')
    parser.add_argument('--model', default='text-embedding-ada-002')
    parser.add_argument('--database_path', default='multiwoz_database')
    args = parser.parse_args()

    docs = []
    embeddings = OpenAIEmbeddings(document_model_name=args.model,
                                  query_model_name=args.model,
                                  openai_api_key=os.environ.get('OPENAI_API_KEY', ''))

    dataset = load_dataset('multi_woz_v22')
    n = 1
    database = MultiWOZDatabase(args.database_path)
    for dialog in dataset['train']:
        if n > 200:
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

            doc = Document(page_content='\n'.join(context),
                           metadata={'domain': f'{domain}',
                                     'state': new_state,
                                     'response': dialog['turns']['utterance'][tn+1],
                                     'database': database_results})
            docs.append(doc)
    faiss_vs = FAISS.from_documents(documents=docs,
                                    embedding=embeddings)
    with open(args.output_faiss_db, 'wb') as f:
        pickle.dump(faiss_vs, f)