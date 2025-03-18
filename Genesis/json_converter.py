import json

with open('/home/matthew0803/domestic-receipts/instructions.json', 'r') as f:
    data = json.load(f)

with open('/home/matthew0803/domestic-receipts/instructions.jsonl', 'w') as f:
    for entry in data:
        f.write(json.dumps(entry) + '\n')
