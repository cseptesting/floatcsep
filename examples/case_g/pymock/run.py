import sys
from pymock import main

"""
Run 'wrapper' function.
 
It receives the input arguments file path (e.g. input/args.txt) from the 
command terminal and passes it to the main() function in pymock/main.py.

1. This file can be run from the command terminal as:
   >>> $ python run.py <args_path>
   with default to <args_path> = input/args.txt

2. It can also be run from a python console if needed.

3. (Advanced) An entry_point was defined, so pymock can also be run as:
   >>> $ pymock <args_path>
   * see setup.cfg and pymock.main:run() for details.
   
"""


def run(default_args='input/args.txt'):

    args = sys.argv  # This reads the arguments given from the terminal

    if len(args) > 1:  # arguments where passed
        print(f'Running using input arguments {args[1]}')
        main.main(*args[1:])

    elif len(args) == 1:  # no args file passed, trying default
        try:
            print(f'Running using default arguments: {default_args}')
            main.main(arg_path=default_args)
        except FileNotFoundError:
            raise FileNotFoundError('Please provide arguments filepath')


if __name__ == '__main__':
    run()
