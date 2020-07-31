"""
Given a CSV containing cleaned AmericanRhetoric data, convert for fine-tuning
of the GPT2, or other model.
"""
import argparse
import csv
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Text, Tuple

import pandas as pd

from loggers import LoggerFactory

# This field_size_limit is needed to import the data CSV
max_int = sys.maxsize
while True:
  try:
    csv.field_size_limit(max_int)
    break
  except OverflowError:
    max_int = int(max_int/10)


class FormatTemplate():
  def __init__(
    self,
    name: Text,
    convert_fn: Callable[..., Callable[[Dict], Text]],
    format_args: Optional[List[Text]],
    eos_token: Text,
    cls_token: Text,
    tag_fn_name: Text
  ) -> None:
    self.name: Text = name
    self.format_args: Optional[List[Text]] = format_args
    self.eos_token: Text = eos_token
    self.cls_token: Text = cls_token
    self.tag_fn_name: Text = tag_fn_name

    self.convert: Callable[[Dict], Text] = convert_fn(
      format_args=self.format_args,
      eos_token=self.eos_token,
      cls_token=self.cls_token,
      make_tag_fn=get_tag_function(self.tag_fn_name)
    )
  
  def __repr__(self) -> Text:
    return "{{name='{}', format_args={}, eos_token={}, cls_token={}, tag_fn_name={}}}".format(
      self.name, self.format_args, repr(self.eos_token), repr(self.cls_token), self.tag_fn_name
    )


def make_tag_v1(header: Text, content: Text) -> Text:
  return '<{}="{}">'.format(header, content)


def make_tag_v2(header: Text, content: Text) -> Text:
  return '<{}= {} >'.format(header, content)


def make_tag_v3(header: Text, content: Text) -> Text:
  return '<|{}|>\n{}'.format(header, content)


ALLOWED_TAG_NAMES: List[Text] = ["v1", "v2", "v3"]
def get_tag_function(tag_name: Text) -> Callable[[Text, Text], Text]:
  return globals()["make_tag_{}".format(tag_name)]


def make_special_token(token: Text) -> Text:
  token = token.strip()
  is_newline = (token == "\n")
  return "{}{}{}".format(
    "" if is_newline else "\n",
    token,
    "" if is_newline else "\n"
  )


def convert_tags_custom(
  format_args: List[Text],
  cls_token: Text,
  eos_token: Text,
  make_tag_fn: Callable[[Text, Text], Text],
  **kwargs: Any
) -> Callable[[Dict], Text]:
  input_headers: List[Text] = format_args[:-1]
  transcript_header: Text = format_args[-1]

  cls_token_out = make_special_token(token=cls_token)
  eos_token_out = make_special_token(token=eos_token)

  def inner_function(d: Dict) -> Text:
    output_row_list: List[Text] = [
      make_tag_fn(header, d[header])
      for header in input_headers
    ]
    output_str: Text = "{features}{cls}{transcript}{eos}".format(
      features="\n".join(output_row_list),
      cls=cls_token_out,
      transcript=d[transcript_header],
      eos=eos_token_out
    )

    return output_str
  
  return inner_function


def convert_tag_all_except_transcript(**kwargs: Any) -> Callable[[Dict], Text]:
  input_headers: List[Text] = [
    "title",
    "speaker",
    "year",
    "summary",
    "transcript"
  ]
  kwargs.pop("format_args", None)
  return convert_tags_custom(
    format_args=input_headers,
    **kwargs
  )


FORMAT_TEMPLATE_FNS: Dict[Text, Callable[..., Callable[[Dict], Text]]] = {
  "tag_all_except_transcript" : convert_tag_all_except_transcript,
  "tags_custom" : convert_tags_custom
}
FORMAT_NAMES: List[Text] = list(FORMAT_TEMPLATE_FNS.keys())
FORMAT_DEFAULT: Text = "tag_all_except_transcript"


def get_format_template(
  format_name: Text,
  format_args: List[Text],
  eos_token: Text,
  cls_token: Text,
  tag_fn_name: Text
) -> FormatTemplate:
  format_convert_fn: Callable[..., Callable[[Dict], Text]] = FORMAT_TEMPLATE_FNS[format_name]
  return FormatTemplate(
    name=format_name,
    convert_fn=format_convert_fn,
    format_args=format_args,
    eos_token=eos_token,
    cls_token=cls_token,
    tag_fn_name=tag_fn_name
  )


def convert_data(
  input_list: List[Dict[Text, Text]],
  output_file_loc: Text,
  format_template: FormatTemplate
) -> None:
  print("\n---Converting data---")
  print("Output location: {}".format(output_file_loc))
  print("Format template: {}".format(format_template))

  num_entries: int = 0

  with open(output_file_loc, "w", encoding="utf-8") as output_file:
    for input_entry in input_list:
      output_txt: Text = format_template.convert(input_entry)
      output_file.write(output_txt)
      num_entries += 1
  
  print("Number of entries: {}".format(num_entries))


def split_and_convert_data(
  input_file_loc: Text,
  output_file_loc: Text,
  format_name: Text,
  format_args: List[Text],
  train_split: float,
  val_split: float,
  random_seed: int,
  eos_token: Text,
  cls_token: Text,
  tag_fn_name: Text
) -> None:
  print("Input csv location: {}".format(input_file_loc))
  print("Output location: {}".format(output_file_loc))
  print("Format name: {}".format(format_name))
  print("Format args: {}".format(format_args))
  print("Training split: {}".format(train_split))
  print("Validation split: {}".format(val_split))
  print("Random seed: {}".format(random_seed))
  print("End of sequence token (eos_token): {}".format(repr(eos_token)))
  print("Classification token (cls_token): {}".format(repr(cls_token)))
  print("Tag function name: {}".format(tag_fn_name))

  # First read the csv
  input_df = pd.read_csv(input_file_loc, quoting=csv.QUOTE_ALL)

  # Compile the tuples of (data split, output file name) to process
  split_pairs: List[Tuple[List, Text]] = []

  if (train_split == 1.0) and (val_split == 0.0):
    # No train-test split, so add to split_pairs
    split_pairs.append((input_df.to_dict("records"), output_file_loc))
  else:
    # Get the output file prefix and extension
    output_file_prefix, output_file_ext = os.path.splitext(output_file_loc)

    if train_split < 1.0:
      # Perform a train-test split
      test_split: float = 1.0 - train_split
      test_df = input_df.sample(
        frac=test_split,
        replace=False,
        random_state=random_seed
      )
      test_output_file_loc: Text = "{}-{}{}".format(
        output_file_prefix,
        "test",
        output_file_ext
      )
      split_pairs.append((test_df.to_dict("records"), test_output_file_loc))

      train_df = input_df.loc[~input_df.index.isin(test_df.index)]
    else:
      # no train-test split
      train_df = input_df
    
    if val_split > 0.0:
      # There is a train-val split

      val_df = train_df.sample(
        frac=val_split,
        replace=False,
        random_state=random_seed
      )
      val_output_file_loc: Text = "{}-{}{}".format(
        output_file_prefix,
        "val",
        output_file_ext
      )
      split_pairs.append((val_df.to_dict("records"), val_output_file_loc))

      train_df = train_df.loc[~train_df.index.isin(val_df.index)]
    
    train_output_file_loc: Text = "{}-{}{}".format(
      output_file_prefix,
      "train",
      output_file_ext
    )
    split_pairs.append((train_df.to_dict("records"), train_output_file_loc))
  
  format_template: FormatTemplate = get_format_template(
    format_name=format_name,
    format_args=format_args,
    eos_token=eos_token,
    cls_token=cls_token,
    tag_fn_name=tag_fn_name
  )

  for split_data, split_output_file_loc in split_pairs:
    convert_data(
      input_list=split_data,
      output_file_loc=split_output_file_loc,
      format_template=format_template
    )


def parse_args() -> argparse.Namespace:
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Convert cleaned CSV into a format that can be used for fine-tuning."
  )
  parser.add_argument(
    "-i", "--input", type=str, required=True,
    help="Input csv location. Should be output of the process_raw_data.py script"
  )

  parser.add_argument(
    "-o", "--output", type=str, required=True,
    help="Location of where to output the converted data"
  )

  parser.add_argument(
    "-l", "--log", type=str, required=False,
    help="(Optional) Location where to output .txt log file"
  )
  
  parser.add_argument(
    "-f", "--format", type=str, nargs="+", default=[FORMAT_DEFAULT],
    help="""The type of format to convert into.
Choose from following, with args if needed: {}""".format(FORMAT_NAMES)
  )

  parser.add_argument(
    "-s", "--split", type=float, required=False, nargs=2, default=[1.0, 0.0],
    help="""The train-val-test split, if desired.
Takes two arguments, of the form 'train val', where both are floats.
The 'train' value is the percentage of the total data to be trained on.
The 'val' value is the percentage of the training data to use for validation.
The remaining (1 - 'train') will be set aside for testing.
Defaults to 'train' = 1 (all training).
    """
  )

  parser.add_argument(
    "-r", "--random", type=int, required=False, default=1234,
    help="The random seed to use. Defaults to 1234"
  )

  parser.add_argument(
    "--eos", type=str, required=False, nargs="?",
    default="\n", const="<|endoftext|>",
    help="""The eos_token to insert in between sequences (speeches).
Can accept a string to use. Spaces are stripped, and it is padded by newlines.
If included without a string (ex: --eos), defaults to <|endoftext|>.
"""
  )

  parser.add_argument(
    "--cls", type=str, required=False, nargs="?",
    default="\n", const="<|cls|>",
    help="""The cls_token to insert in between the features and the transcripts.
Can accept a string to use. Spaces are stripped, and it is padded by newlines.
If included without a string (ex: --cls), defaults to <|cls|>.
"""
  )

  parser.add_argument(
    "-t", "--tag", type=str, required=False, default="v3",
    help="Specify the type of tagging to use. Choose from {}".format(
      ALLOWED_TAG_NAMES
    ), choices=ALLOWED_TAG_NAMES
  )

  args = parser.parse_args()

  split_train, split_val = args.split
  if (split_train < 0) or (split_train > 1):
    parser.error("Train split {} is invalid.".format(split_train))
  if (split_val < 0) or (split_val > 1):
    parser.error("Val split {} is invalid.".format(split_val))
  
  return args


def main() -> None:
  args: argparse.Namespace = parse_args()
  
  log_file_loc = args.log if args.log else ""
  with LoggerFactory(log_file_loc) as logger_factory:
    logger_factory.set_loggers()

    print("{}\n\n".format(args))

    split_and_convert_data(
      input_file_loc=args.input,
      output_file_loc=args.output,
      format_name=args.format[0],
      format_args=args.format[1:],
      train_split=args.split[0],
      val_split=args.split[1],
      random_seed=args.random,
      eos_token=args.eos,
      cls_token=args.cls,
      tag_fn_name=args.tag
    )


if __name__ == "__main__":
  main()
