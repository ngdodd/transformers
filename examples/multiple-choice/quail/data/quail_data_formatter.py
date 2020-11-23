# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:22:16 2020

@author: ngd
"""
import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to unformatted custom dataset.')
    parser.add_argument('--output_file', type=str, help='Path to quail formatted output file location.')
    args = parser.parse_args()
    
    # Read in the current dataset and shuffle it
    with open(args.input_file, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        random.shuffle(data)
        
    # Process each entry in our dataset and write out to a jsonl file formatted similarly to quail.
    with open(args.output_file, mode='w', encoding='utf-8') as f:
        for id_, entry in enumerate(data):
            # Modify entry components to match quail format
            entry['id'] = "custom_{}".format(id_)
            entry['context'] = entry.pop('Context')
            entry['question'] = entry.pop('Question')
            entry['question_type'] = entry.pop('Reasoning type')
            entry['answers'] = [entry['Choices']['A'], entry['Choices']['B'], entry['Choices']['C'], entry['Choices']['D']]
            entry['correct_answer_id'] = str(ord(entry['Answer'].capitalize()) - 65)
            
            # TODO: For answerable questions, randomly change one of the incorrect
            # answer options to be "not enough information"
            """
            if entry['question_type'] != 'Unanswerable':
                nei_index = random.choice([i for i in range(len(entry['answers'])) if i != int(entry['correct_answer_id'])])
                entry['answers'][nei_index] = "not enough information"
            """
            
            # Rename coreference question type to match Quail's task names
            entry['question_type'] = 'Character_identity' if entry['question_type'] == 'coreference' else entry['question_type']
            
            # Delete duplicates
            del entry['Choices']
            del entry['Answer']
            
            # Write the json line to file
            json.dump(entry, f)
            f.write('\n')
        
if __name__ == "__main__":
    main()