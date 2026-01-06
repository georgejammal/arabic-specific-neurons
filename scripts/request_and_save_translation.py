import html
import json 
import requests 
import time 
from datasets import load_dataset 

# config 
DATASET_NAME = "agentlans/high-quality-english-sentences" 
SPLIT = "train" 
NUM_SAMPLES = 1000 
OUTPUT_PATH = "../datasets/en_1k.json"

API_KEY = "AIzaSyAjJTIvknoLQaDDyVG5bom2enF8Jj5gzLQ"
TRANSLATE_URL = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"

# load and save 1k dataset samples from hf 
def save_first_1k_english():
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    samples = [] 

    for i in range(NUM_SAMPLES):
        samples.append({"id": i, "text": ds[i]["text"]})
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


    print(f"Saved {NUM_SAMPLES} samples to {OUTPUT_PATH}")
   
   
# translate using google translate api 
def translate_batch(sentences, target_lang): 
    payload = {
        "q": sentences,
        "target": target_lang,
    }
    response = requests.post(TRANSLATE_URL, json=payload) 
    if response.status_code != 200:
        raise Exception(f"Translation API error: {response.text}") 

    return [item['translatedText'] for item in response.json()['data']['translations']]


def translate_1k_sentences(lang_code, output_filename, batch_size=50):
    with open("../datasets/en_1k.json", "r", encoding="utf-8") as f:
        en_data = json.load(f)

    texts = [x["text"] for x in en_data]
    translated = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_trans = translate_batch(batch, lang_code)

        for j, t in enumerate(batch_trans):
            translated.append({
                "id": i + j,
                "text": html.unescape(t),
            })

        time.sleep(0.1)

    with open(f"../datasets/{output_filename}", "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    print(f"Saved {output_filename}")



if __name__ == "__main__":
    save_first_1k_english()

    translate_1k_sentences("ar", "ar_1k.json")
    translate_1k_sentences("he", "he_1k.json")
    translate_1k_sentences("zh-CN", "zh_1k.json")
   
