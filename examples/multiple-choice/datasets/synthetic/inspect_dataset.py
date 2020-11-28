import json
import string
import numpy as np
from scipy import stats
from datasets import load_dataset
import matplotlib.pyplot as plt

def get_dataset(path, split='train'):
    dataset = load_dataset('json', data_files=path)[split]
    print("\n{}\n".format(dataset))
    return dataset

def inspect_dataset_types(dataset):
    types = {}
    for entry in dataset:
        q_type = entry['question_type']
        if q_type in types:
            types[q_type] += 1
        else:
            types[q_type] = 1
    print("Types: {}".format(types))
    
def plot_word_counts(dataset):
    word_counts = {}
    for entry in dataset:
        q_type = entry['question_type']
        word_count = len(entry['context'].translate(str.maketrans('', '', string.punctuation)).split())
        if q_type in word_counts:
            word_counts[q_type].append(word_count)
        else:
            word_counts[q_type] = [word_count]
    
    plt.title("Question Type Word Counts")
    types_ = sorted(list(word_counts.keys()))
    for type_ in types_:
        if type_ == "Character_identity":
            plt.hist(word_counts[type_], bins=100, edgecolor='k', label=type_)
        else:
            plt.hist(word_counts[type_], bins=50, edgecolor='k', label=type_)
    plt.minorticks_on()
    plt.legend()
    
def catch_duplicates(dataset):
    samples = {}
    duplicates = {}
    for entry in dataset:
        key = "{}{}{}".format(entry['question'],entry['context'],str(entry['answers']))
        id_ = entry['id']
        if key in samples:
            if id_ in duplicates:
                duplicates[id_] += 1
            else:
                duplicates[id_] = 1
        else:
            samples[key] = entry['id']
    return duplicates

def filter_context_length(dataset, threshold):
    context_filtered = {}
    for entry in dataset:
        word_count = len(entry['context'].translate(str.maketrans('', '', string.punctuation)).split())
        if word_count > threshold:
            context_filtered[entry['id']] = 1
    return context_filtered

def filter_question_type(dataset, question_type):
    question_type_filtered = {}
    for entry in dataset:
        if entry['question_type'] == question_type:
            question_type_filtered[entry['id']] = 1
    return question_type_filtered

def save_dataset(dataset, path, to_remove=[]):
    with open(path, 'w', encoding='utf-8') as w:
        for entry in dataset:
            if entry['id'] not in to_remove:
                w.write(json.dumps(entry) + '\n')
    
def main():
    path = 'synthetic_with_nei.jsonl'
    dataset = get_dataset(path)
    inspect_dataset_types(dataset)
    plot_word_counts(dataset)
    duplicates = catch_duplicates(dataset)
    context_filtered = filter_context_length(dataset, threshold=400)
    type_filtered = filter_question_type(dataset, question_type='Sequential')
    to_remove = list(duplicates.keys()) + list(context_filtered.keys()) + list(type_filtered.keys())
    save_dataset(dataset, path="{}_without_sequential_cleaned.jsonl".format(path[:path.rfind('.jsonl')]), to_remove=to_remove)
    
if __name__ == "__main__":
    main()
