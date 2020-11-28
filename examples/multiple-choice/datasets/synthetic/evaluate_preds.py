import json
import argparse

def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--key', required=True, help='Key JSON file.')
    parser.add_argument('--pred', required=True, help='Model predictions.')
    parser.add_argument('--out_file', '-o', metavar='eval.json', help='Write accuracy metrics to file (default is stdout).')
    args = parser.parse_args()

    with open(args.key, 'r', encoding='utf-8') as g, open(args.pred, 'r', encoding='utf-8') as p:
        gold = json.load(g)
        pred = json.load(p)

    type_lookup = {}
    for type_, subset in gold.items():
        for id_, label in subset.items():
            type_lookup[id_] = type_    

    accuracies = {key: [] for key in gold.keys()}
    n_questions = len(pred)
    for id_, ans in pred.items():
        q_type = type_lookup[id_]
        accuracies[q_type].append(int(gold[q_type][id_]==ans))
    total_accuracy = sum([sum(values) for values in accuracies.values()])/n_questions
    accuracies = {key: 100*sum(values)/len(values) for key, values in accuracies.items()}
    accuracies['Total'] = total_accuracy*100
    
    with open(args.out_file, 'w', encoding='utf-8') as w:
        json.dump(accuracies, w, indent=2)

if __name__== "__main__":
    main()