# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""


import logging
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils_multiple_choice import (
    MultipleChoiceDataset, 
    Split, 
    breakdown_reasoning_accuracy, 
    processors
)


logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


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
    quail_dev_key_path: Optional[str] = field(
        default=None,
        metadata={"help": "The location of the quail dev key to use during evaluation."}
    )
    custom_training_filename: Optional[str] = field(
        default=None,
        metadata={"help": "Filename of the custom training set to use during model training."}
    )
    custom_eval_filename: Optional[str] = field(
        default=None,
        metadata={"help": "File name for custom dev set to evaluate the model on."}
    )
    custom_dev_key_path: Optional[str] = field(
        default=None,
        metadata={"help": "The location of the dev key to use with the custom dev set."}
    )
    eval_on_training_set: Optional[bool] = field(
        default=False,
        metadata={"help": "Flag controlling whether or not to do an additional round of evaluation \
                                 on the training set, to analyze cases of overfitting."}
    )
                  
def evaluate_model(
    trainer: Trainer,
    eval_dataset: MultipleChoiceDataset,
    eval_name: str,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    dev_key: Optional[str] = None,
) -> None:
    """
    Parameters
    ----------
    trainer : Trainer
        Huggingface trainer object that will be used to obtain model predictions.
    eval_dataset : MultipleChoiceDataset
        Featurized dataset which will be used to evaluate the model.
    eval_name : str
        A descriptive name for this evaluation round used to distinguish results.
    training_args : TrainingArguments
        Set of training related arguments specified when running the script.
    data_args : DataTrainingArguments
        Set of data related arguments specified when running the script.
    dev_key : Optional[str]
        Filename of the dev key used to break down accuracy across reasoning types.
    """
    logger.info("*** Evaluate ***")
    predictions = trainer.predict(eval_dataset)
    
    # Get mcq preds, labels, and metrics
    mcq_preds = np.argmax(predictions.predictions, axis=1)
   
    # Create result dictionaries to be dumped to output json files
    ids = [feature.example_id for feature in eval_dataset.features]
    mcq_results = {id: pred for id, pred in zip(ids, mcq_preds.tolist())}
    
    # Predictions to be used in eval script
    output_preds_file = os.path.join(training_args.output_dir, "{}_preds.json".format(eval_name))
    if trainer.is_world_master():
        with open(output_preds_file, 'w', encoding='utf-8') as writer:
            json.dump(mcq_results, writer, separators=(',', ':'), sort_keys=True, indent=4)
            
    # Write prediction metrics to file
    output_metrics_file = os.path.join(training_args.output_dir, "{}_metrics.json".format(eval_name))
    if trainer.is_world_master():
        with open(output_metrics_file, 'w', encoding='utf-8') as writer:
            json.dump(predictions.metrics, writer, separators=(',', ':'), sort_keys=True, indent=4)
    
    if trainer.is_world_master():
        logger.info("\n\n\n\n***** {} Results *****".format(eval_name))
        for key, value in predictions.metrics.items():
            logger.info("  %s = %s", key, value)
            
    breakdown_reasoning_accuracy(
        trainer=trainer,
        dev_key=dev_key,
        preds_map=mcq_results,
        save_path=os.path.join(training_args.output_dir, "{}_results.json".format(eval_name))
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    # Get datasets
    training_file_name = data_args.custom_training_filename if data_args.custom_training_filename else None
    train_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            file_name=training_file_name,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
            
    # Evaluation
    if training_args.do_eval:
        evaluate_model(
            trainer=trainer, 
            eval_dataset=eval_dataset,
            eval_name="Quail_Dev_Eval",
            training_args=training_args,
            data_args=data_args,
            dev_key=data_args.quail_dev_key_path
        )
        
        if data_args.custom_eval_filename:
            custom_eval_dataset = MultipleChoiceDataset(
                data_dir=data_args.data_dir,
                file_name=data_args.custom_eval_filename,
                tokenizer=tokenizer,
                task=data_args.task_name,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.dev,
            )
            evaluate_model(
                trainer=trainer,
                eval_dataset=custom_eval_dataset,
                eval_name='Custom_Dev_Eval',
                training_args=training_args,
                data_args=data_args,
                dev_key=data_args.custom_dev_key_path
            )
            
        if data_args.eval_on_training_set:
            if not training_args.do_train:
                train_dataset = MultipleChoiceDataset(
                    data_dir=data_args.data_dir,
                    tokenizer=tokenizer,
                    task=data_args.task_name,
                    max_seq_length=data_args.max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                    mode=Split.train,
                    file_name=training_file_name,
                )
                
            evaluate_model(
                trainer=trainer,
                eval_dataset=train_dataset,
                eval_name='Train_Eval',
                training_args=training_args,
                data_args=data_args,
                dev_key=data_args.custom_dev_key_path
            )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
