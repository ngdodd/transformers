# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:03:15 2020

@author: nickg
"""
import os
import json
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    
@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]
    
class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
class QuailProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        print("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        print("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        print("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "challenge.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]
    
    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines
        
    def _create_examples(self, lines, type):
        examples = [
            InputExample(
                example_id=qa_entry["id"],
                question=qa_entry["question"],
                contexts=[qa_entry["context"], 
                          qa_entry["context"],
                          qa_entry["context"],
                          qa_entry["context"]],
                endings=[qa_entry["answers"][0],
                         qa_entry["answers"][1],
                         qa_entry["answers"][2],
                         qa_entry["answers"][3]],
                label=qa_entry["correct_answer_id"]
            )
            for qa_entry in [json.loads(line) for line in lines]
        ]
        
        return examples

"""    
qp = QuailProcessor()

train_examples = qp.get_train_examples('.')
print("{} training examples created".format(len(train_examples)))
print("\nTrain example sample:\n{}\n".format(train_examples[0]))

test_examples = qp.get_test_examples('.')
print("{} test examples created".format(len(test_examples)))
print("\nTest example sample:\n{}\n".format(test_examples[0]))

dev_examples = qp.get_dev_examples('.')
print("{} dev examples created".format(len(dev_examples)))
print("\nDev example sample:\n{}\n".format(dev_examples[0]))
"""
