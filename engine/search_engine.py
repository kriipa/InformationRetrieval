import json
import os
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)

class SimpleSearchEngine:
    def __init__(self, data):
        self.data = data
        self.inverted_index = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'\w+')

    def _preprocess(self, text):
        if not text or not isinstance(text, str):
            return []

        tokens = self.tokenizer.tokenize(text.lower())
        tokens = [t for t in tokens if t not in self.stop_words]
        if not tokens:
            return []

        try:
            pos_tags = pos_tag(tokens)
        except Exception:
            pos_tags = [(t, 'NN') for t in tokens]

        def _get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        lemmas = set()
        for token, tag in pos_tags:
            wn_pos = _get_wordnet_pos(tag)
            try:
                lemma_primary = self.lemmatizer.lemmatize(token, pos=wn_pos)
            except Exception:
                lemma_primary = token

            try:
                lemma_verb = self.lemmatizer.lemmatize(token, pos=wordnet.VERB)
            except Exception:
                lemma_verb = token

            try:
                lemma_noun = self.lemmatizer.lemmatize(token, pos=wordnet.NOUN)
            except Exception:
                lemma_noun = token

            lemmas.add(token)
            lemmas.add(lemma_primary)
            lemmas.add(lemma_verb)
            lemmas.add(lemma_noun)

            if token.endswith('ing') and len(token) > 4:
                lemmas.add(token[:-3])
            if token.endswith('ed') and len(token) > 3:
                lemmas.add(token[:-2])

        cleaned = [l for l in lemmas if l and l not in self.stop_words]
        return list(cleaned)

    def _build_index(self):
        print("Building weighted inverted index from source data...")
        index = defaultdict(list)
        for doc_id, doc in enumerate(self.data):
            title = doc.get('title', '') or ''
            abstract = doc.get('abstract', '') or ''
            kws = doc.get('keywords', []) or []
            if isinstance(kws, list):
                keywords = ' '.join(kws)
            elif isinstance(kws, str):
                keywords = kws
            else:
                keywords = ''
            subject = doc.get('subject_areas', '')
            if isinstance(subject, list):
                subject_areas = ' '.join(subject)
            elif isinstance(subject, str):
                subject_areas = subject
            else:
                subject_areas = ''
            authors_list = doc.get('authors', []) or []
            author_names_list = []
            for a in authors_list:
                if isinstance(a, dict):
                    author_names_list.append(a.get('name', ''))
                elif isinstance(a, str):
                    author_names_list.append(a)
            author_names = ' '.join([n for n in author_names_list if n])
            weighted_text = ' '.join([title] * 3 + [keywords] * 3 + [author_names] * 2 + [abstract] * 2 + [subject_areas])
            tokens = self._preprocess(weighted_text)
            for token in set(tokens):
                index[token].append(doc_id)
        self.inverted_index = dict(index)
        print("Weighted inverted index built.")

    def build(self, force_rebuild=False):
        index_path = './data/inverted_index.json'
        os.makedirs('./data', exist_ok=True)
        if os.path.exists(index_path) and not force_rebuild:
            print(f"Loading inverted index from {index_path}...")
            with open(index_path, 'r') as f:
                self.inverted_index = json.load(f)
            print("Inverted index loaded.")
        else:
            self._build_index()
            print(f"Saving inverted index to {index_path}...")
            with open(index_path, 'w') as f:
                json.dump(self.inverted_index, f, indent=4)
            print("Inverted index saved.")

    def search(self, query):
        if self.inverted_index is None:
            raise RuntimeError("Index is not built. Please call .build() before searching.")
        query_tokens = self._preprocess(query)
        if not query_tokens:
            return []
        matching_docs_set = set(self.inverted_index.get(query_tokens[0], []))
        for token in query_tokens[1:]:
            matching_docs_set.intersection_update(self.inverted_index.get(token, []))
        results = [self.data[doc_id] for doc_id in sorted(list(matching_docs_set))]
        return results

if __name__ == "__main__":
    try:
        with open('./data/publications_data.json', 'r') as f:
            publication_data = json.load(f)
    except FileNotFoundError:
        print("Error: publications_data.json not found. Please place it in the 'data' directory.")
        exit()
    engine = SimpleSearchEngine(data=publication_data)
    engine.build()
    search_query = "account"
    print(f"\nSearching for: '{search_query}'")
    results = engine.search(search_query)
    if results:
        print(f"Found {len(results)} matching documents:")
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.get('title', 'Untitled')}")
    else:
        print("No documents found containing all query terms.")
