import json

import pandas as pd
with open('../dataset/MELD_train_efr.json') as f:
    task3_data = json.load(f)

# print(task3_data["utterances"])
df = pd.DataFrame(columns=('speakers','utterances', 'emotions', 'triggers'))

index = 0
for dialog in task3_data:
    # print(dialog)
    for (speaker, utterance, emotion, trigger_value) in zip(dialog["speakers"], dialog["utterances"], dialog["emotions"], dialog["triggers"]):
        # print(utterance, emotion, trigger_value)
        df.loc[index] = [speaker, utterance, emotion, trigger_value]
        index += 1

# print(df)
X = pd.DataFrame(df, columns=["speakers", "emotions", "utterances"])

print(X.head(3))
