from datasets import load_dataset
import json , random
ds = load_dataset("openai/gsm8k", "main", split="test")
ids = list(range(len(ds)))
random.Random(0).shuffle(ids)
pools = {"proposal": ids[:100], "dev": ids[100:300], "audit": ids[300:700]}

assert not (set(pools["proposal"]) & set(pools["dev"])) and set(pools["proposal"]) & set(pools["audit"]) and set(pools["dev"]) & set(pools["audit"])
json.dump(pools,open("config/pools_gsm8k.json","w"))

def gold(item):
    return float(item["answer"].split("####")[1].strip().replace(",", ""))