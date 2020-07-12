import argparse
from corpus import constants
import csv
import json
import os
from common import utils


def get_statistics(parsed_dir, save_file):
    congress_nums = [113, 114, 115, 116]
    rows = []
    for congress_num in congress_nums:
        file_prefix = 'CHRG-' + str(congress_num)
        json_files = utils.get_files_starting_with(parsed_dir, file_prefix)
        for json_file in json_files:
            with open(os.path.join(parsed_dir, json_file), 'r') as json_data:
                hearing = json.load(json_data)
            hearing_id = hearing[constants.HEARING_ID]
            turns = hearing[constants.HEARING_TURNS]
            num_turns = len(turns)
            witnesses = []
            for turn in turns:
                if turn[constants.TURN_SPEAKER_TYPE] == constants.TURN_SPEAKER_TYPE_WITNESS:
                    witnesses.append(turn[constants.TURN_SPEAKER])
            num_witnesses = len(set(witnesses))
            num_questions = len([turn for turn in turns if turn[constants.TURN_IS_QUESTION]])
            rows.append([congress_num, hearing_id, num_turns, num_questions, num_witnesses])
    with open(save_file, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(['congress num', 'hearing id', 'num turns', 'num questions', 'num_witnesses'])
        csvwriter.writerows(rows)
    print('Wrote csv file with stats to ', save_file, '.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Corpus Stats 1.0')
    parser.add_argument('--parsed_dir', type=str, help='Path to parsed congressional hearings')
    parser.add_argument('--save_file', type=str, help='CSV file to save stats')
    args = parser.parse_args()
    print("Parser args: ", args)
    get_statistics(args.parsed_dir, args.save_file)
