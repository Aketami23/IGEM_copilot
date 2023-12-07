import openai
import time
from vector_store import VectorStore  
from make_index import get_size 

PROMPT = """
## Example
{text}
==
## requirement
{input}
""".strip()

MAX_PROMPT_SIZE = 8190
RETURN_SIZE = 4000
MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds

def search_query(input_str, index_file):
    PROMPT_SIZE = get_size(PROMPT)
    rest = MAX_PROMPT_SIZE - RETURN_SIZE - PROMPT_SIZE
    input_size = get_size(input_str)
    if rest < input_size:
        raise RuntimeError("too large input!")
    rest -= input_size

    vs = VectorStore(index_file, create_if_not_exist=False)
    samples = vs.get_sorted(input_str)

    # Exclude results with Title ending in "_0
    filtered_samples = [sample for sample in samples if not sample[2].endswith("_0")]

    to_use = []
    used_title = []
    for _sim, body, title in filtered_samples[:2]:  # Get top 7 results
        if title in used_title:
            continue
        size = get_size(body)
        if rest < size:
            break
        to_use.append(body)
        used_title.append(title)
        rest -= size
    # 選ばれたパーツを表示
    print("Selected Parts based on Similarity:")
    for part in to_use:
        print(f"- {part}")

    text = "\n\n".join(to_use)
    prompt = PROMPT.format(input=input_str, text=text)

    retries = 0
    while retries < MAX_RETRIES:
        try:
            print("\nTHINKING...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",###gpt-3.5-turbo-1106###gpt-4-1106-preview
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=RETURN_SIZE,
                temperature=0.4,
            )
            break
        except openai.error.RateLimitError as e:
            print(f"RateLimitError encountered. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            RETRY_DELAY *= 2
            retries += 1
    else:
        print("Failed to get a response after multiple retries.")
        return

    # show question and answer
    content = response['choices'][0]['message']['content']
    print("\nANSWER:")
    print(f">>>> {input_str}")
    print(">", content)

if __name__ == "__main__":
    query = "Tell us about your original project and its feasibility using horizontal and diffuse thinking with examples."  # ここに検索したいクエリを入力
    index_file = "project_index.pickle"
    search_query(query, index_file)
