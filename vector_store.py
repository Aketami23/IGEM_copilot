import time
import json
import tiktoken
import openai
import pickle
import numpy as np
from tqdm import tqdm
import dotenv
import os

def split_data_by_part_id(raw_data):
    split_data_by_lines = raw_data.split('",')
    split_data_by_parts = [part.split('\t') for part in split_data_by_lines]
    return split_data_by_parts

BLOCK_SIZE = 1000
EMBED_MAX_SIZE = 80000000000

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
enc = tiktoken.get_encoding("cl100k_base")

def get_size(text):
    return len(enc.encode(text))

def embed_text(text, sleep_after_success=1):
    text = text.replace("\n", " ")
    tokens = enc.encode(text)
    if len(tokens) > EMBED_MAX_SIZE:
        text = enc.decode(tokens[:EMBED_MAX_SIZE])

    while True:
        try:
            res = openai.Embedding.create(
                input=text,##input=[text]でやってた
                model="text-embedding-ada-002")
            time.sleep(sleep_after_success)
        except Exception as e:
            print(e)
            time.sleep(1)
            continue
        break

    return res["data"][0]["embedding"]

def update_from_igem_parts(json_file, out_index, in_index=None):

    
    if in_index is not None:
        cache = pickle.load(open(in_index, "rb"))
    else:
        cache = None

    vs = VectorStore(out_index)
    data = json.load(open(json_file, encoding="utf8"))

    for p in tqdm(data["pages"]):
        title = p["title"]
        for i, line in enumerate(p["lines"]):
            part_info = line
            unique_title = f"{title}_{i}"  # 各行に独自の識別子を付与
            
            # 入力データの確認
            print(f"Text to be embedded: {part_info}")

            # エラーメッセージの詳細表示
            try:
                vs.add_record(part_info, unique_title, cache)
            except Exception as e:
                print(f"Exception details: {str(e)}")
                time.sleep(1)

    vs.save()


class VectorStore:
    def __init__(self, name, create_if_not_exist=True):
        self.name = name
        try:
            self.cache = pickle.load(open(self.name, "rb"))
        except FileNotFoundError as e:
            if create_if_not_exist:
                self.cache = {}
            else:
                raise

    def add_record(self, body, title, cache=None):
        if cache is None:
            cache = self.cache
        if body not in cache:
            self.cache[body] = (embed_text(body), title)
        elif body not in self.cache:
            self.cache[body] = cache[body]

    def get_sorted(self, query):
        q = np.array(embed_text(query, sleep_after_success=0))
        buf = []
        for body, (v, title) in tqdm(self.cache.items()):
            buf.append((q.dot(v), body, title))
        buf.sort(reverse=True)
        return buf

    def save(self):
        pickle.dump(self.cache, open(self.name, "wb"))

if __name__ == "__main__":
    JSON_FILE = "from_scrapbox/igemparts (1).json"
    INDEX_FILE = "igemindex4.pickle"
    update_from_igem_parts(JSON_FILE, INDEX_FILE)
    
    # Sample search query
    vs = VectorStore(INDEX_FILE, create_if_not_exist=False)
    query = "Your search query here"
    results = vs.get_sorted(query)
    
    for score, body, title in results:
        print(f"Score: {score}, Body: {body}, Title: {title}")
