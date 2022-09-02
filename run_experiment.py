# Local modules
import sys
from fecsep import run, models

if __name__ == "__main__":

    # name = 'team'
    # folder = './models/TEAM'
    # team = models.Model('team', folder, None, None, zenodo_id=6289795)
    # team.stage(format='hdf5')
    # parse command line arguments
    if len(sys.argv) == 1:
        print('Usage: python run_experiment <test_date>')
        print("\ttest_date (required) : format='%Y-%m-%dT%H:%M:%S'")
        sys.exit(-1)
    #
    # # run gefe with command line arguments
    run(sys.argv[1])




