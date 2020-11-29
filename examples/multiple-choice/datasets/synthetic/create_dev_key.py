import json
import argparse
from datasets import load_dataset
        
def write_dev_key(dataset, types, path, verbose):    
    # Map data to type map with dataset filters
    type_map = {}
    for type_ in types:
        type_map[type_] = dataset.filter(lambda e: e['question_type'] == type_)
    
    # Get keys for each dataset
    key_map = {type_: {entry['id']: int(entry['correct_answer_id']) for entry in subset} for type_, subset in type_map.items() }

    if verbose:    
        print("\ntype_map: {}".format(type_map))
        print("\nKey: {}".format(key_map))

    with open(path, 'w', encoding='utf-8') as w:
       json.dump(key_map, w)
       
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to input dataset location.')
    parser.add_argument('--out_file', type=str, help='Path to dev key output file location.')
    parser.add_argument('--verbose', type=bool, default=False, help='Turn on verbose mode to see the key map creation printed to console.')
    args = parser.parse_args()
    
    dataset = load_dataset('json', data_files=args.in_file)['train']
    if args.verbose:
        print(dataset)
    
    # Do a pass through the dataset to get the question types used
    question_types = {}
    for entry in dataset:
        if entry['question_type'] not in question_types:
            question_types[entry['question_type']] = 1

    write_dev_key(dataset, list(question_types.keys()), args.out_file, args.verbose)
    
if __name__ == "__main__":
    main()