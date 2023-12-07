import openai
import time
from vector_store import VectorStore 
from make_index import get_size 

PROMPT = """
## Given parts.
{text}
## What we are looking for.
{input}
Tell me how to modify the given parts, along with the part number(BBa_).
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
    for _sim, body, title in filtered_samples[:4]:  # Get top 7 results
        if title in used_title:
            continue
        size = get_size(body)
        if rest < size:
            break
        to_use.append(body)
        used_title.append(title)
        rest -= size
    # Show selected parts
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
                model="gpt-4-0613",###gpt-3.5-turbo-1106###gpt-4-1106-preview
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=RETURN_SIZE,
                temperature=0.2,
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
    query = "CodingPart:BBa_K4410003Designed by: Ruyi Shi   Group: iGEM22_Worldshaper-HZBIOX   (2022-09-21)KanRThis part is KanR gene, which is a kanamycin resistance gene from pKD4 vector. The gene encodes aminoglycoside phosphotransferase from Tn5，which confers resistance to neomycin, kanamycin, and G418 (Geneticin®).Therefore, only the bacteria carrying KanR gene can grow on the culture dish containing kanamycin, so as to achieve the goal of rapid screening of engineering bacteria. In our project, we used KanR gene to replace EcN1917 argR gene to eliminate argR feedback repression.Sequence and Features"  # ここに検索したいクエリを入力
    index_file = "igemindex4.pickle" 
    search_query(query, index_file)
