import json

with open('../dataset/MELD_train_efr.json', 'r') as f:
    dataset = json.load(f)

statistic_triggers = {}
statistic_all_triggers = {}
nr_of_triggers = 0
for dialog in dataset:
    for index, trigger in enumerate(dialog['triggers']):
        if trigger == 1:
            nr_of_triggers += 1
            if dialog['speakers'][index] not in statistic_triggers.keys():
                statistic_triggers[dialog['speakers'][index]] = 1
            else:
                statistic_triggers[dialog['speakers'][index]] += 1

            if dialog['emotions'][index] not in statistic_triggers.keys():
                statistic_triggers[dialog['emotions'][index]] = 1
            else:
                statistic_triggers[dialog['emotions'][index]] += 1

            if dialog['speakers'][index] + " | " + dialog['emotions'][index] not in statistic_triggers.keys():
                statistic_triggers[dialog['speakers'][index] + " | " + dialog['emotions'][index]] = 1
            else:
                statistic_triggers[dialog['speakers'][index] + " | " + dialog['emotions'][index]] += 1
                
        if dialog['speakers'][index] not in statistic_all_triggers.keys():
            statistic_all_triggers[dialog['speakers'][index]] = 1
        else:
            statistic_all_triggers[dialog['speakers'][index]] += 1

        if dialog['emotions'][index] not in statistic_all_triggers.keys():
            statistic_all_triggers[dialog['emotions'][index]] = 1
        else:
            statistic_all_triggers[dialog['emotions'][index]] += 1

        if dialog['speakers'][index] + " | " + dialog['emotions'][index] not in statistic_all_triggers.keys():
            statistic_all_triggers[dialog['speakers'][index] + " | " + dialog['emotions'][index]] = 1
        else:
            statistic_all_triggers[dialog['speakers'][index] + " | " + dialog['emotions'][index]] += 1

final_statistics = {}
for stats in statistic_triggers:
    # print("|||||||||| " + stats + " ||||||||||")
    # print("Triggerability: " + str(statistic_triggers[stats]/statistic_all_triggers[stats]*100) + "%")
    # print("Triggered: " + str(statistic_triggers[stats]/nr_of_triggers*100) + "%")
    final_statistics[stats] = statistic_triggers[stats]/statistic_all_triggers[stats]*100

final_statistics = {k: v for k, v in sorted(final_statistics.items(), key=lambda item: item[1], reverse=True)}

for stats in final_statistics:
    print("|||||||||| " + stats + " ||||||||||")
    print("Triggerability: " + str(final_statistics[stats]) + " %")
    print("Triggered: " + str(final_statistics[stats]/nr_of_triggers*100) + " %")

        
        
