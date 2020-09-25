import os
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from typing import List

from cookiecutter.main import cookiecutter
from transformers.commands import BaseTransformersCLICommand

from ..utils import logging
import torch
import curses
from curses.textpad import Textbox, rectangle

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ChecklistManager:
    class ChecklistLayer(OrderedDict):

        index = 0
        all_layers: List["ChecklistManager.ChecklistLayer"] = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, enabled=True, **kwargs)

            # Important to keep track of the index of each layer in the ordered dict to enable/disable them afterwards.
            self.index = ChecklistManager.ChecklistLayer.index
            ChecklistManager.ChecklistLayer.all_layers.append(self)
            ChecklistManager.ChecklistLayer.index += 1

        def get_layers(self):
            return {key: value.get_layers() for key, value in self.items() if
                    type(value) is ChecklistManager.ChecklistLayer and value['enabled']}

    def __init__(self, checklist_name: str, keys: List):
        ChecklistManager.ChecklistLayer.index = 0
        ChecklistManager.ChecklistLayer.all_layers = []
        self.checklist_name = checklist_name

        # When layers are folded their indices should not be counted
        self.layer_indices_to_skip = set()

        def regroup(state_dict_keys: List[str]):
            lists = [key.split('.') for key in state_dict_keys]
            layer = ChecklistManager.ChecklistLayer()
            for state_dict_list in lists:
                if state_dict_list[0] not in layer.keys():
                    layer[state_dict_list[0]] = []

                layer[state_dict_list[0]].append('.'.join(state_dict_list[1:]))

            for key, value in layer.items():
                if isinstance(value, list) and len(value) > 0:
                    if len(value) == 1:
                        if value[0] == '':
                            layer[key] = ChecklistManager.ChecklistLayer()
                            # layer[key]["torch _buffer"] = ChecklistManager.ChecklistLayer()
                        elif value[0] == 'weight':
                            layer[key] = ChecklistManager.ChecklistLayer()
                            # layer[key]["nn.Embeddings"] = ChecklistManager.ChecklistLayer()
                    elif len(value) == 2:
                        if value[0] in ('weight', 'bias') and value[1] in ('weight', 'bias'):
                            if 'LayerNorm' in key or 'layer_norm' in key:
                                layer[key] = ChecklistManager.ChecklistLayer()
                                # layer[key]["torch.nn.LayerNorm"] = ChecklistManager.ChecklistLayer()
                            else:
                                layer[key] = ChecklistManager.ChecklistLayer()
                                # layer[key]["torch.nn.Linear"] = ChecklistManager.ChecklistLayer()
                    else:
                        layer[key] = regroup(value)
            return layer

        self.architecture = regroup(keys)
        self.num_layers = len(self.ChecklistLayer.all_layers)

    @staticmethod
    def to_str(layer, nesting_level=1):
        return "".join(
            [
                nesting_level * "    "
                + "└──"
                + key
                + f'    [{"x" if layer[key]["enabled"] else " "}]'
                + "\n"
                + (ChecklistManager.to_str(layer[key], nesting_level=nesting_level + 1) if len(layer[key]) > 0 and layer[key]['enabled'] else "")
                for key in layer.keys() if key != 'enabled'
            ]
        )

    def __str__(self):
        return f"{self.checklist_name}" + "\n" + self.to_str(self.architecture)

    def toggle_layer(self, index):
        # Folded layers should be skipped
        for layer in self.layer_indices_to_skip:
            if layer <= index:
                index += 1
        is_enabled = self.ChecklistLayer.all_layers[index]['enabled']

        if is_enabled:
            # If the layer is currently enabled, disable all the layers inside it when folding it
            def recurrent_disable(layer: ChecklistManager.ChecklistLayer):
                for key, value in layer.items():
                    if type(value) == ChecklistManager.ChecklistLayer:
                        value['enabled'] = False
                        # Add the layer indices to the set
                        self.layer_indices_to_skip.add(value.index)
                        recurrent_disable(value)

            self.ChecklistLayer.all_layers[index]['enabled'] = False
            recurrent_disable(self.ChecklistLayer.all_layers[index])
        else:
            # If the layer is disabled, enable this specific layer
            self.ChecklistLayer.all_layers[index]['enabled'] = True
            for key, value in ChecklistManager.ChecklistLayer.all_layers[index].items():
                # Remove the layer indices from the set
                if type(value) == ChecklistManager.ChecklistLayer:
                    self.layer_indices_to_skip.remove(value.index)

    def get_layers(self):
        return self.ChecklistLayer.all_layers[0].get_layers()

    def get_layer(self, index: int):
        return self.ChecklistLayer.all_layers[index]


class CursesCLI:
    def __init__(self):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.scrollok(True)

        self.stdscr.addstr(0, 0, "Welcome to the model creation page!")
        self.stdscr.addstr(1, 0, "This tool is here to help you add a model to the HuggingFace API.")
        self.stdscr.addstr(2, 0, "Please complete the following information to get you set up.")

    @staticmethod
    def clear_lines(win_or_pad, _from, _to):
        for i in range(_from, _to):
            win_or_pad.move(i, 0)
            win_or_pad.clrtoeol()

    def get_message(self, question, default=None, choices=None):
        # CLear lines where there might have been text previously
        self.clear_lines(self.stdscr, _from=4, _to=8)
        self.stdscr.addstr(4, 0, question + (f" (Example: {default})" if default is not None else ""))

        # Create windows and rectangle
        if choices is not None:
            curses.curs_set(False)
            width = sum([len(choice) for choice in choices]) + 4
            rectangle(self.stdscr, 5, 0, 1 + 5 + 1, 1 + width + 1)
            selected_index = 0

            def draw_choices():
                for i, choice in enumerate(choices):
                    highlight_current = selected_index == i
                    offset = sum([len(previous_choice) + 2 for previous_choice in choices[:i]])
                    self.stdscr.addstr(6, 2 + offset, choice, curses.A_REVERSE if highlight_current else curses.A_NORMAL)

            draw_choices()
            self.stdscr.refresh()

            key = self.stdscr.getkey()
            while key != '\n':
                if key == "KEY_LEFT" and selected_index > 0:
                    selected_index -= 1
                elif key == "KEY_RIGHT" and selected_index < len(choices) - 1:
                    selected_index += 1

                draw_choices()
                key = self.stdscr.getkey()

            curses.curs_set(True)
            return choices[selected_index]
        else:
            editable_window = curses.newwin(1, 100, 6, 1)
            rectangle(self.stdscr, 5, 0, 1 + 5 + 1, 1 + 100 + 1)
            self.stdscr.refresh()

            # Create textbox in window and let user edit it
            box = Textbox(editable_window)
            box.edit()

            # Gather and return the values
            result = box.gather()
            return result if result != "" else default

    def get_checkpoint_mapping(self, model_name, state_dict):
        architecture = ChecklistManager(model_name + 'Model', list(state_dict.keys()))
        return self.get_checklist_result(architecture)

    def get_checklist_result(self, architecture: ChecklistManager):
        # CLear lines where there might have been text previously
        self.clear_lines(self.stdscr, _from=4, _to=7)
        self.stdscr.addstr(4, 0, "Please select which items you would like to have generated by the template.")
        self.stdscr.refresh()
        max_y, max_x = self.stdscr.getmaxyx()

        # Create a pad as pads can be larger than windows
        pad = curses.newpad(50, 100)
        pad_y_max = max_y - 2
        pad_y_min = 5
        pad_scroll = 0

        # Create and print the complete state dict on the pad
        num_layers = architecture.num_layers

        nested_state_dict = str(architecture)
        split_nested_state_dict = nested_state_dict.split('\n')
        pad.addstr(nested_state_dict)
        pad.refresh(0, 0, pad_y_min, 0, pad_y_max, max_x)

        self.stdscr.move(pad_y_min, len(split_nested_state_dict[0]))

        key = self.stdscr.getkey()
        while key != '\n':
            y, x = self.stdscr.getyx()
            if key == "KEY_UP":
                if y > pad_y_min:
                    self.stdscr.move(y - 1, len(split_nested_state_dict[y - pad_y_min + pad_scroll - 1]) - 2)
                elif pad_scroll > 0:
                    pad_scroll -= 1
                    pad.refresh(pad_scroll, 0, pad_y_min, 0, pad_y_max, max_x)
                    self.stdscr.move(y, len(split_nested_state_dict[y - pad_y_min + pad_scroll]) - 2)
            elif key == "KEY_DOWN":
                if y < pad_y_max - 1 and (y - pad_y_min + pad_scroll + 1 < len(split_nested_state_dict) - 1):
                    self.stdscr.move(y + 1, len(split_nested_state_dict[y - pad_y_min + pad_scroll + 1]) - 2)
                elif (pad_scroll + pad_y_max - pad_y_min) < len(split_nested_state_dict) - 1:
                    pad_scroll += 1
                    pad.refresh(pad_scroll, 0, pad_y_min, 0, pad_y_max, max_x)
                    self.stdscr.move(y, len(split_nested_state_dict[y - pad_y_min + pad_scroll]) - 2)
            elif key == ' ' and not (y == pad_y_min and pad_scroll == 0):
                targeted_string_index = y - pad_y_min + pad_scroll
                architecture.toggle_layer(targeted_string_index)
                nested_state_dict = str(architecture)
                split_nested_state_dict = nested_state_dict.split('\n')
                self.clear_lines(pad, 0, num_layers)
                pad.addstr(0, 0, str(architecture))
                pad.refresh(pad_scroll, 0, pad_y_min, 0, pad_y_max, max_x)
                self.stdscr.move(y, x)

            key = self.stdscr.getkey()

        return architecture.get_layers()


def add_new_model_command_factory(args):
    return AddNewModelCommand()


class AddNewModelCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("add-new-model")
        download_parser.set_defaults(func=add_new_model_command_factory)

    def run(self):

        from cookiecutter.main import cookiecutter

        path_to_transformer_root = Path(__file__).parent.parent.parent.parent
        path_to_cookiecutter = path_to_transformer_root / "templates" / "cookiecutter"

        model_name = "DistilBert"
        authors = "Lysandre Debut"
        checkpoint_identifier = "distilbert-base-uncased",

        model_heads = ["Question Answering", "Token Classification"]

        model_outputs = ',\n    '.join([''.join(model_head.split(" ")) + 'ModelOutput' for model_head in model_heads])

        # call to header
        cookiecutter(str(path_to_cookiecutter), extra_context={
            "model_name": model_name,
            "checkpoint_identifier": checkpoint_identifier,
            "model_outputs": model_outputs,
            "file_part_name": "header"
        }, no_input=True, directory="misc/header")

        # call to pretrained and base model
        cookiecutter(str(path_to_cookiecutter), extra_context={
            "model_name": model_name,
            "checkpoint_identifier": checkpoint_identifier,
            "model_outputs": model_outputs,
            "file_part_name": "pretrained_and_base_model"
        }, no_input=True, directory="pretrained_and_base_model")

        # call to model heads
        for model_head in model_heads:
            cookiecutter(str(path_to_cookiecutter), extra_context={
                "model_name": model_name,
                "checkpoint_identifier": checkpoint_identifier,
                "model_outputs": model_outputs,
                "file_part_name": model_head
            }, no_input=True, directory=f"model_heads/{'_'.join(model_head.lower().split())}")

        import sys
        sys.exit()

        cookiecutter(str(path_to_cookiecutter), extra_context={})

        directory = [directory for directory in os.listdir() if "cookiecutter-template-" in directory[:22]][0]

        cookiecutter()

        cli = CursesCLI()

        model_name = cli.get_message("What is your model name?", default="DistilBERT")
        authors = cli.get_message("Who are the authors?", default="The HuggingFace team")
        encoder_decoder = cli.get_message("Is your model seq2seq?", choices=["Encoder-decoder", "Single-stack"])
        checkpoint = cli.get_message("Do you have an checkpoint for your model (can be TF1, TF2 or PyTorch)?", choices=["Yes", "No"])

        if checkpoint == "Yes":
            checkpoint_path = cli.get_message(
                f"Please enter the absolute path to your checkpoint.",
                default=f"/Users/jik/Workspaces/python/transformers/albert/pytorch_model.bin"
            )
            print(checkpoint)
            state_dict = torch.load(checkpoint_path)
            architecture = cli.get_checkpoint_mapping(model_name, state_dict)

        heads_checklist = ChecklistManager(
            'Model with Heads',
            [
                'Masked LM',
                'Causal LM',
                'Question Answering',
                'Token Classification',
                'Sequence Classification',
                'Multiple Choice'
            ]
        )
        heads = cli.get_checklist_result(heads_checklist)

        inputs_checklist = ChecklistManager(
            'Inputs accepted',
            [
                "Position IDs",
                "Token type IDs (also known as Segment IDs)"
            ]
        )
        accepted_inputs = cli.get_checklist_result(inputs_checklist)

        added_functionalities_checklist = ChecklistManager(
            'Added functionalities',
            [
                "Feedforward chunking",
                "Gradient checkpointing",
                "Head masking",
                "Prunable layers"
            ]
        )
        added_functionalities = cli.get_checklist_result(added_functionalities_checklist)
        tokenizer_algorithm = cli.get_message("What is your tokenizer algorithm?", choices=["Wordpiece", "BPE", "SentencePiece"])






        # Get the nesting level


        # import sys
        # sys.exit(1)
        # directories = [directory for directory in os.listdir() if "cookiecutter-template-" in directory[:22]]
        # if len(directories) > 0:
        #     raise ValueError(
        #         "Several directories starting with `cookiecutter-template-` in current working directory. "
        #         "Please clean your directory by removing all folders startign with `cookiecutter-template-` or "
        #         "change your working directory."
        #     )
        #
        # path_to_transformer_root = Path(__file__).parent.parent.parent.parent
        # path_to_cookiecutter = path_to_transformer_root / "templates" / "cookiecutter"
        # checkpoint_path = "/Users/jik/Workspaces/python/transformers/albert/pytorch_model.bin"
        # state_dict = torch.load(checkpoint_path)
        # cookiecutter(str(path_to_cookiecutter), extra_context={"template_nesting_level": get_nested_level_from_pytorch_state_dict(state_dict)})
        #
        # directory = [directory for directory in os.listdir() if "cookiecutter-template-" in directory[:22]][0]
        #
        # import shutil
        #
        # model_name = directory[22:]
        # lowercase_model_name = model_name.lower()
        #
        # print(directory, model_name, lowercase_model_name)
        #
        # shutil.move(
        #     f"{directory}/configuration_{lowercase_model_name}.py",
        #     f"{path_to_transformer_root}/src/transformers/configuration_{lowercase_model_name}.py",
        # )
        #
        # shutil.move(
        #     f"{directory}/modeling_{lowercase_model_name}.py",
        #     f"{path_to_transformer_root}/src/transformers/modeling_{lowercase_model_name}.py",
        # )
        #
        # os.rmdir(directory)

#
# class LayerUtils:
#
#     lowercase_model_name = '{{cookiecutter.lowercase_model_name}}'
#     model_name = '{{cookiecutter.model_name}}'
#     uppercase_model_name = '{{cookiecutter.uppercase_model_name}}'
#     checkpoint_identifier = '{{cookiecutter.checkpoint_identifier}}'
#
#     @staticmethod
#     def convert_to_camel_case(string_value):
#         return ''.join([val.capitalize() for val in string_value.split('_')])
#
#     def create_model(self, architecture):
#         for key,value
#         total_model = architecture
#
#
#     @staticmethod
#     def create_layer(layer_name, layer_type=None):
#         layer = """
#         class {model_name}{classname}(nn.Module):
#             def __init__():
#                 ""
#         """.format(
#             model_name=LayerUtils.model_name,
#             classname=LayerUtils.convert_to_camel_case(layer_name)
#         )
#
#         return layer
