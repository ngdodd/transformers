import os
import json
import inspect
import argparse
from typing import Dict, List, Optional
from enum import Enum
from filelock import FileLock
from dataclasses import dataclass, field

import torch
import tqdm
import datasets
import numpy as np
from torch.utils.data.dataset import Dataset
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

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
        reasoning_label: (Optional) string. The reasoning type label for the
        example. Specified for train and dev samples, but not for test samples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]
    reasoning_label: Optional[str]


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
    reasoning_label: Optional[int]
    
class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    
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
    def __init__(self):
        self.reasoning_type_dict = {'Character_identity': 0, 'Causality': 1, 'Event_duration': 2, 
                                    'Subsequent_state': 3, 'Factual': 4, 'Belief_states': 5, 
                                    'Entity_properties': 6, 'Unanswerable':7, 'Temporal_order':8}
        
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
    
    def get_reasoning_labels(self):
        return list(self.reasoning_type_dict.keys())
    
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
                label=qa_entry["correct_answer_id"],
                reasoning_label=qa_entry["question_type"]
            )
            for qa_entry in [json.loads(line) for line in lines]
        ]
        
        return examples

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}

processors = {"quail": QuailProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"quail", 4}

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    reasoning_label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=True,
            )
            """
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )
            """
            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )
        
        reasoning_label_map = {reasoning_label: i for i, reasoning_label in enumerate(reasoning_label_list)} if reasoning_label_list != None else None
        reasoning_label = reasoning_label_map[example.reasoning_label] if reasoning_label_map != None else None
        
        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
                reasoning_label=reasoning_label
            )
        )

    for f in features[:2]:
        print("*** Example ***")
        print("feature: %s" % f)

    return features


class MultipleChoiceDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        processor = processors[task]()

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(max_seq_length),
                task,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                print(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                print(f"Creating features from dataset file at {data_dir}")
                label_list = processor.get_labels()
                reasoning_label_list = processor.get_reasoning_labels()
                if mode == Split.dev:
                    examples = processor.get_dev_examples(data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)
                print("Training examples: %s", len(examples))
                self.features = convert_examples_to_features(
                    examples,
                    label_list,
                    reasoning_label_list,
                    max_seq_length,
                    tokenizer,
                )
                print("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

"""************************************************************************"""
"""************************************************************************"""

def test_signature(model, dataset):
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += ["label", "label_ids"]
    
    columns = [k for k in signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    dataset.set_format(type=dataset.format["type"], columns=columns)
    print("Ignored columns: {}".format(ignored_columns))

def main():
    data_dir = "quail/data"
    model_name = "microsoft/deberta-base"
    max_seq_length = 32
    
    qp = QuailProcessor()
    train_examples = qp.get_train_examples(data_dir)
    print("{} training examples created".format(len(train_examples)))
    
    test_examples = qp.get_test_examples(data_dir)
    print("{} test examples created".format(len(test_examples)))
    
    dev_examples = qp.get_dev_examples(data_dir)
    print("{} dev examples created".format(len(dev_examples)))
    
    config = AutoConfig.from_pretrained(
            model_name,
            num_labels=4,
            reasoning_types=9,
            finetuning_task="quail",
            cache_dir=None,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=None,
        )
    
    model = AutoModelForMultipleChoice.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=config,
            cache_dir=None,
        )
    
    # Get datasets
    train_dataset = (
        MultipleChoiceDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            task="quail",
            max_seq_length=max_seq_length,
            overwrite_cache=False,
            mode=Split.train,
        )
    )
    
    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            task="quail",
            max_seq_length=max_seq_length,
            overwrite_cache=False,
            mode=Split.dev,
        )
    )

    # Initialize our Trainer
    training_args = TrainingArguments(
        output_dir="./quail_out",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=25,
        num_train_epochs=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    results = {}
    print("*** Evaluate ***")
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    ids = [feature.example_id for feature in eval_dataset.features]
    results = {id: pred for id, pred in zip(ids, preds.tolist())}

    output_preds_file = os.path.join(training_args.output_dir, "preds.json")
    if trainer.is_world_master():
        with open(output_preds_file, 'w', encoding='utf-8') as writer:
            json.dump(results, writer, separators=(',', ':'), sort_keys=True, indent=4)

    output_labels_file = os.path.join(training_args.output_dir, "labels.json")
    if trainer.is_world_master():
        with open(output_labels_file, 'w', encoding='utf-8') as writer:
            json.dump(predictions.label_ids.tolist(), writer, separators=(',', ':'), sort_keys=True, indent=4)

    output_metrics_file = os.path.join(training_args.output_dir, "metrics.json")
    if trainer.is_world_master():
        with open(output_metrics_file, 'w', encoding='utf-8') as writer:
            json.dump(predictions.metrics, writer, separators=(',', ':'), sort_keys=True, indent=4)
    
    result = trainer.evaluate()

    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            print("***** Eval results *****")
            for key, value in result.items():
                print("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

            results.update(result)
    print("Eval results: {}".format(results))

    
if __name__ == "__main__":
    main()
    