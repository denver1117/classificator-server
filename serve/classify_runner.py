"""
Classificator runner script
"""

import argparse
import datetime
import traceback
from classificator.classify import Classificator

def main(args):
    # Run the classificator
    clf = Classificator(config_loc=args.config_loc)
    try:
        clf.choose_model()
    except Exception as e:
        # Write the traceback to log
        with open("{0}".format(clf.log_name), "a") as f:
            now = datetime.datetime.now()
            f.write("{0} - Runtime Error\n".format(now))
            tb = traceback.format_exc()
            f.write(tb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run classification')
    parser.add_argument('-c', '--config',
                        dest='config_loc',
                        default='config.json',
                        help='input configuration location')
    args = parser.parse_args()
    main(args)

