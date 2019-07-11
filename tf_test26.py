import argparse
import os
import sys

current_path = os.path.dirname(os.path.realpath("abcd"))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

print(FLAGS)
print(unparsed)
