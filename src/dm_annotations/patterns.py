import json

with open("resources/modality-patterns.json") as f:
    modality_patterns = json.loads(f.read())

# TODO ものの as 文中接続表現
with open("resources/connectives-patterns.json") as f:
    connectives_patterns = json.loads(f.read())

with open("resources/connectives-regexes.json") as f:
    connectives_regexes = json.loads(f.read())

if __name__ == "__main__":
    print(modality_patterns)
    print(connectives_patterns)
    print(connectives_regexes)
