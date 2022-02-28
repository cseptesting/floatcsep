# Local modules
import sys
from gefe import run

if __name__ == "__main__":

    # parse command line arguments
    if len(sys.argv) == 1:
        print('Usage: python run_experiment <test_date>')
        print("\ttest_date (required) : format='%Y-%m-%dT%H:%M:%S'")
        sys.exit(-1)

    # run experiment with command line arguments
    run(sys.argv[1])




