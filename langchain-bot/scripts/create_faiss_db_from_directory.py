import argparse
import pickle
import glob
import os

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_texts_dir')
    parser.add_argument('--output_faiss_db')
    parser.add_argument('--model', default='text-embedding-ada-002')
    args = parser.parse_args()

    index_data = []
    docs = []
    embeddings = OpenAIEmbeddings(document_model_name=args.model,
                                  query_model_name=args.model,
                                  openai_api_key=os.environ.get('OPENAI_API_KEY', ''))

    for fn in glob.glob(f'{args.input_texts_dir}/*.txt'):
        doc_name = fn.split('/')[-1].split('.')[0]
        with open(fn, 'rt') as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = Document(page_content=line,
                                   metadata={'topic': f'{doc_name}'})
                    docs.append(doc)
    faiss_vs = FAISS.from_documents(documents=docs,
                                    embedding=embeddings)
    with open(args.output_faiss_db, 'wb') as f:
        pickle.dump(faiss_vs, f)