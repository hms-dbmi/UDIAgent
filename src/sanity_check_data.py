from datasets import load_from_disk

data = load_from_disk('/n/netscratch/mzitnik_lab/Lab/dlange/data/dqvis_training')


def print_content(i, only_data = False):
    content = data['train'][i]['messages'][0]['content']
    query = data['train'][i]['messages'][1]['content']
    if only_data:
        content = content[135:-102]
    print(query)
    print(content)

print_content(0)

# for i in range(5):
#     print_content(i)