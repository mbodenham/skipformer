import math
import torch
import wandb

from torch.utils.data import Dataset
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm
from transformers import Trainer, TrainerCallback
from typing import Dict, List, Optional
from scipy import stats


from torch.profiler import record_function, ProfilerActivity
import math
# import torchprof
# pip install torchprof


class SkipformerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            if isinstance(model, DistributedDataParallel) and model.training:
                if model.module.transformer.attention_window:
                    losses = torch.zeros(len(model.module.transformer.attention_window_layers))
                    for l, layer in enumerate(model.module.transformer.attention_window_layers):

                        loss += layer.attention_window.get_loss() * 0.0001
                        # losses[l] = layer.attention_window.get_loss()
                    # loss += losses.mean() * 0.0001

            elif model.training:
                if model.transformer.attention_window:
                    losses = torch.zeros(len(model.transformer.attention_window_layers))
                    for l, layer in enumerate(model.transformer.attention_window_layers):

                        loss += layer.attention_window.get_loss() * 0.0001
                        # losses[l] = layer.attention_window.get_loss()
                    # loss += losses.mean() * 0.0001

            return (loss, outputs) if return_outputs else loss

    def compute_latency(
            self,
            repetitions: int = 100,
            attention_window: bool = True,
            masked: bool = True,
            profile: bool = False,
        ) -> float:


        # model = self._wrap_model(self.model, training=False)
        model = self.model
        model.eval()


        print(f"***** Collecting Latency *****")
        print(f"\tRepetitions: {repetitions}")
        print(f"\tAttention Window: {attention_window}")

        device = self.args.device
        N = 1024
        with torch.no_grad():
            if isinstance(model, DistributedDataParallel) or isinstance(model, DataParallel):
                attention_layers = model.module.transformer.attention_window_layers
            else:
                attention_layers = model.transformer.attention_window_layers

            windows = []
            for layer in attention_layers:
                if not attention_window:
                    # layer.attention_window.max_window_size = N
                    layer.attention_window.current_val[0] = 1.
                    if masked:
                        layer.attention_window.masked = True
                    else:
                        layer.attention_window.masked = False
                z = layer.attention_window.init_mask()
                print(z, math.ceil(z*N+16), layer.attention_window.masked)
                windows.append(math.ceil(z*N+16))


            print(sum(windows)/len(windows), max(windows), min(windows))
            # exit()

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timings = torch.zeros(repetitions, device=device)

            inputs = {'input_ids': torch.randint(low=0, high=50257, size=[1, N], device=device),
                      'attention_mask': torch.ones([1, N], device=device),
                      'labels': torch.randint(low=0, high=50257, size=[1, N], device=device)}

            # warm up
            for _ in tqdm(range(repetitions)):
                outputs = model(**inputs)

            for rep in tqdm(range(repetitions)):
                starter.record(torch.cuda.current_stream(device))
                outputs = model(**inputs)
                ender.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)

                timings[rep] = starter.elapsed_time(ender)
                starter.elapsed_time(ender)

        timings = timings.cpu().detach().numpy()
        z_score = stats.zscore(timings)
        timings = timings[(-3<z_score) & (z_score<3)]
        avg_time = timings.mean()

        if profile:
            with torch.no_grad():
                with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
                    # with record_function("model_inference"):
                    for _ in range(1):
                        model(**inputs)
                # with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
                #     model(**inputs)

            # print(prof.display(show_events=False))


            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            prof.export_chrome_trace("trace.json")

        print(f"***** Completed *****")
        print(f"\tLatency (ms): {avg_time}")

        return avg_time


class LogWindowSize(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model.training:
            for layer in model.transformer.attention_window_layers:
                layer.attention_window.clamp_param()

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model.training:
            if logs.get('loss'):
                logs['perplexity'] = math.exp(logs['loss'])
            if model.transformer.attention_window:
                sizes = []
                loss = 0
                losses = torch.zeros(len(model.transformer.attention_window_layers))
                for l, layer in enumerate(model.transformer.attention_window_layers):
                    for head, current_size in enumerate(layer.attention_window.get_current_size().cpu().detach()):
                        sizes.append(current_size.item())
                        logs[f'Window Size All/Layer{l}/Head{head}'] = current_size.item()

                    loss += layer.attention_window.get_loss().cpu().detach().item() * 0.0001
                    # losses[l] = layer.attention_window.get_loss()
                # loss = losses.mean().cpu().detach().item() * 0.0001


                logs['window_loss'] = loss
                logs['Window Size/Average'] = sum(sizes) // len(sizes)
                logs['Window Size/Maximum'] = int(max(sizes))
                logs['Window Size/Minimum'] = int(min(sizes))
