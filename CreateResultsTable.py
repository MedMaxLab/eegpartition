import argparse
import sys
sys.path.append('..')
from AllFnc.utilities import gather_results

if __name__ == '__main__':

    help_d = """
    CreateResultsTable gathers the results stored in all pickle files into a
    unique Pandas DataFrame, which is stored in a csv file with
    default name 'ResultsTable.csv'.
    A custom name can be given in input as well.
    
    Example:
    
    $ python3 CreateResultsTable.csv -n CustomName.csv
    
    """
    parser = argparse.ArgumentParser(description=help_d)
    parser.add_argument(
        "-n",
        "--name",
        dest      = "filename",
        metavar   = "csv filename",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = None,
        help      = """
        The results filename. it should include the .csv extension at the end. 
        However, the function can handle its absence. 
        """,
    )

    # basically we overwrite the dataPath if something was given
    args = vars(parser.parse_args())
    FilenameInput = args['filename']
    gather_results(save = True, filename = FilenameInput)
    