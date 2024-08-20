
# Task-oriented LLM Bot

Code accompaniment for the paper [Are Large Language Models All You Need for Task-Oriented Dialogue?](https://aclanthology.org/2023.sigdial-1.21/) (Hudeček & Dušek, SIGDIAL 2023).

### Installation

Use your Conda/Virtualenv and simply run:
```
pip install -r requirements.txt
```


### Usage

1. You need to create the FAISS vector store DB:
```
python create_faiss_db.py --output_faiss_db mwoz_db.pkl --total 10 --dataset multiwoz
```

2. To use GPT-3.5, add your OpenAI API key into the file `openai_key.txt`.

3. Edit the `run_interact.sh` file in case you want to include your own CUDA/Python etc. setup.

4. You can now run the interactive bot with default settings like this:
```
./run_interact.sh
```

Alternatively, you can run `interact.py` directly and change the parameters.
* `num_examples` changes the number of few-shot examples in the prompt
* `context_size` changes the length of the dialogue context passed to the bot
* `goals_trial.json` is just a trial file to try out user goal messages, you can use MultiWOZ 2.1's `data.json` directly (this is just a small sample of that file).
* Other required files (database, ontology) are directly taken from MultiWOZ.


### Cite us

If you use this code in your work, please cite this paper:

```
@inproceedings{hudecek_are_2023,
	address = {Prague, Czechia},
	title = {Are {Large} {Language} {Models} {All} {You} {Need} for {Task}-{Oriented} {Dialogue}?},
	url = {https://aclanthology.org/2023.sigdial-1.21},
	doi = {10.18653/v1/2023.sigdial-1.21},
	booktitle = {Proceedings of the 24th {Annual} {Meeting} of the {Special} {Interest} {Group} on {Discourse} and {Dialogue}},
	publisher = {Association for Computational Linguistics},
	author = {Hudeček, Vojtěch and Dusek, Ondrej},
	month = sep,
	year = {2023},
	pages = {216--228},
}
```


### Acknowledgements

This work was funded by the European Union (ERC, NG-NLG, 101039303), project TL05000236 AI asistent pro žáky a učitele co-financed by the Technological Agency of the Czech Republic within the ÉTA 5 Programme, and Charles University project SVV 260 698. It used resources of the LINDAT/CLARIAH-CZ Research Infrastructure (Czech Ministry of Education, Youth, and Sports project No. LM2018101).
