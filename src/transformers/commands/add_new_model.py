import os
from argparse import ArgumentParser
from pathlib import Path

from cookiecutter.main import cookiecutter
from transformers.commands import BaseTransformersCLICommand

from ..utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_nesting_levels():
    nesting_levels = [
        {"{}Encoder": {}, "{}Embeddings": {}},
        {
            "{}Encoder": {
                "{}Layer": {},
                "...": {},
            },
            "{}Embeddings": {"word_embeddings": {}, "position_embeddings": {}, "token_type_embeddings": {}},
        },
        {
            "{}Encoder": {
                "{}Layer": {"{}Attention": {}, "{}Intermediate": {}, "{}Output": {}},
                "...": {},
            },
            "{}Embeddings": {"word_embeddings": {}, "position_embeddings": {}, "token_type_embeddings": {}},
        },
    ]

    def to_str(dictionary, nesting_level=1):
        return "".join(
            [
                nesting_level * "\t"
                + "└──"
                + key
                + "\n"
                + (to_str(dictionary[key], nesting_level=nesting_level + 1) if len(dictionary[key]) > 0 else "")
                for key in dictionary.keys()
            ]
        )

    return [("{}Model" + "\n" + to_str(level)).replace("{}", "{{cookiecutter.modelname}}") for level in nesting_levels]


def add_new_model_command_factory(args):
    return AddNewModelCommand()


class AddNewModelCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("add-new-model")
        download_parser.set_defaults(func=add_new_model_command_factory)

    def run(self):
        import curses
        from curses.textpad import Textbox, rectangle

        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.start_color()
        stdscr.keypad(True)
        stdscr.scrollok(True)

        stdscr.addstr(0, 0, "Welcome to the model creation page!")
        stdscr.addstr(1, 0, "This tool is here to help you add a model to the HuggingFace API.")
        stdscr.addstr(2, 0, "Please complete the following information to get you set-up.")

        def get_message(question, choices=None):
            # CLear line where there might have been text previously
            stdscr.move(4, 0)
            stdscr.clrtoeol()
            stdscr.addstr(4, 0, question)

            # Clear existing rectangles
            stdscr.move(5, 0)
            stdscr.clrtoeol()
            stdscr.move(6, 0)
            stdscr.clrtoeol()
            stdscr.move(7, 0)
            stdscr.clrtoeol()

            # Create windows and rectangle
            if choices is not None:
                width = sum([len(choice) for choice in choices]) + 4
                editwin = curses.newwin(1, width, 6, 1)
                rectangle(stdscr, 5, 0, 1 + 5 + 1, 1 + width + 1)

                for i, choice in enumerate(choices):
                    offset = sum([len(previous_choice) + 2 for previous_choice in choices[:i]])

                    stdscr.attron(curses.A_REVERSE | curses.color_pair(1))
                    stdscr.addstr(6, 2 + offset, choice)
                    stdscr.attroff(curses.A_REVERSE | curses.color_pair(0))

                stdscr.refresh()

                stdscr.getch()
            else:
                editwin = curses.newwin(1, 50, 6, 1)
                rectangle(stdscr, 5, 0, 1 + 5 + 1, 1 + 50 + 1)
                stdscr.refresh()

                # Create textbox in window and let user edit it
                box = Textbox(editwin)
                box.edit()

                # Gather and return the values
                return box.gather()

        modelname = get_message("What is your model name? (Example: DistilBERT)")
        authors = get_message("Who are the authors? (Example: The HuggingFace team)")
        checkpoint = get_message("Do you have an available checkpoint?", choices=["Yes", "No"])
        encoder_decoder = get_message("Is your model an encoder-decoder or simply an encoder?", choices=["Encoder-decoder", "Encoder"])

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
        # cookiecutter(str(path_to_cookiecutter), extra_context={"template_nesting_level": get_nesting_levels()})
        #
        # directory = [directory for directory in os.listdir() if "cookiecutter-template-" in directory[:22]][0]
        #
        # import shutil
        #
        # modelname = directory[22:]
        # lowercase_modelname = modelname.lower()
        #
        # print(directory, modelname, lowercase_modelname)
        #
        # shutil.move(
        #     f"{directory}/configuration_{lowercase_modelname}.py",
        #     f"{path_to_transformer_root}/src/transformers/configuration_{lowercase_modelname}.py",
        # )
        #
        # shutil.move(
        #     f"{directory}/modeling_{lowercase_modelname}.py",
        #     f"{path_to_transformer_root}/src/transformers/modeling_{lowercase_modelname}.py",
        # )
        #
        # os.rmdir(directory)
