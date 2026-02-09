import gc
import logging
import numpy as np
import os
import pickle
import tempfile
import torch
import time

from datetime import datetime

from llm_processes.hf_api import get_model_and_tokenizer
from llm_processes.parse_args import llm_map, parse_command_line
from llm_processes.run_llm_process import run_llm_process

from .base import Baseline

class LLMPForecaster(Baseline):
    __version__ = "0.0.4"

    def __init__(self, llm_type, use_context=True, dry_run=False):
        self.llm_type = llm_type
        self.use_context = use_context

        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self.tmpdir.name
        self.input_data_path = f"{self.output_dir}/input_data.tmp"
        self.experiment_name = "llmp_runner"
        self.output_data_path = f"{self.output_dir}/{self.experiment_name}.pkl"

        self.llmp_args = {
            "--llm_type": llm_type,
            "--data_path": self.input_data_path,
            "--forecast": "true",
            "--autoregressive": "true",
            "--output_dir": self.output_dir,
            "--experiment_name": self.experiment_name,
            "--num_samples": None,
            "--mode": "sample_only",
        }

        if not dry_run:
            logging.info("Loading model and tokenizer...")
            try:
                self.model, self.tokenizer = get_model_and_tokenizer(
                    llm_path=None, llm_type=self.llmp_args["--llm_type"]
                )
            except KeyError:
                raise ValueError(
                    f"Model type {self.llmp_args['--llm_type']} not supported. Options are: {llm_map.keys()}"
                )
        else:
            logging.info("Dry run: Model and tokenizer not loaded.")
            self.model, self.tokenizer = None, None

    def _prepare_data(self, task_instance):
        logging.info("Preparing data for LLMP...")
        llmp_data = {}
        past_time = task_instance.past_time[task_instance.past_time.columns[-1]]
        future_time = task_instance.future_time[task_instance.future_time.columns[-1]]
        llmp_data["x_train"] = past_time.index.strftime("%Y-%m-%d %H:%M:%S").values
        llmp_data["x_test"] = future_time.index.strftime("%Y-%m-%d %H:%M:%S").values
        llmp_data["x_true"] = np.hstack((llmp_data["x_train"], llmp_data["x_test"]))
        llmp_data["x_ordering"] = {
            t: int(datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp())
            for t in llmp_data["x_true"]
        }
        llmp_data["y_train"] = past_time.values
        llmp_data["y_test"] = future_time.values
        llmp_data["y_true"] = np.hstack((llmp_data["y_train"], llmp_data["y_test"]))
        with open(self.llmp_args["--data_path"], "wb") as f:
            pickle.dump(llmp_data, f)

    def _load_results(self):
        logging.info("Loading results from LLMP...")
        with open(self.output_data_path, "rb") as f:
            results = pickle.load(f)
            samples = np.array(results["y_test"]).transpose()
        return samples

    def _make_prompt(self, task_instance):
        self.llmp_args["--prefix"] = prompt

    def __call__(self, task_instance, n_samples):
        starting_time = time.time()
        logging.info("Forecasting with LLMP...")
        self._prepare_data(task_instance)

        logging.info("Preparing prompt...")
        if self.use_context:
            self._make_prompt(task_instance)
        else:
            if "--prefix" in self.llmp_args:
                del self.llmp_args["--prefix"]

        self.llmp_args["--num_samples"] = str(n_samples)

        logging.info("Running LLM process...")
        llmp_args = parse_command_line(
            [item for pair in self.llmp_args.items() for item in pair]
        )
        start_inference = time.time()
        run_llm_process(args=llmp_args, model=self.model, tokenizer=self.tokenizer)
        end_inference = time.time()

        samples = self._load_results()
        extra_info = {
            "inference_time": end_inference - start_inference,
            "total_time": time.time() - starting_time,
        }
        return samples[:, :, None], extra_info

    @property
    def cache_name(self):
        args_to_include = ["llm_type", "use_context"]
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )
