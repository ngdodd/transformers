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
        print(key_map)

    with open(path, 'w', encoding='utf-8') as w:
       json.dump(key_map, w)
       
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to input dataset location.')
    parser.add_argument('--out_file', type=str, help='Path to dev key output file location.')
    parser.add_argument('--verbose', type=bool, default=False, help='Turn on verbose mode to see the key map creation printed to console.')
    args = parser.parse_args()
    
    types = ["Causality", "Sequential", "Character_identity", "Unanswerable"]
    dataset = load_dataset('json', data_files=args.in_file)['train']
    if args.verbose:
        print(dataset)

    write_dev_key(dataset, types, args.out_file, args.verbose)
    
if __name__ == "__main__":
    main()