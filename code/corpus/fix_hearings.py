import argparse
from corpus import constants
import csv
import json
import os
from common import utils


def save_empty_speakers(parsed_hearings_dir, congress_num, save_dir):
    print("Getting")
    empty_speakers = get_empty_speakers(parsed_hearings_dir, congress_num)
    if not empty_speakers:
        print("No empty speakers!")
        return
    # write to csv file:
    csv_file = os.path.join(save_dir, str(congress_num) + "_empty_speakers.tsv")
    with open(csv_file, 'w') as f:
        csvwriter = csv.writer(f, delimiter='\t')
        csvwriter.writerow([constants.HEARING_ID, constants.TURN_SPEAKER])
        csvwriter.writerows(empty_speakers)


def get_empty_speakers(parsed_hearings_dir, congress_num):
    empty_speakers_info = []
    file_prefix = 'CHRG-' + str(congress_num)
    json_files = utils.get_files_starting_with(parsed_hearings_dir, file_prefix)
    for json_file in json_files:
        with open(os.path.join(parsed_hearings_dir, json_file), 'r') as json_data:
            hearing = json.load(json_data)
        hearing_id = hearing[constants.HEARING_ID]
        turns = hearing[constants.HEARING_TURNS]
        for turn in turns:
            speaker = turn[constants.TURN_SPEAKER]
            if [hearing_id, speaker] not in empty_speakers_info and not turn[constants.TURN_SPEAKER_FULL]:
                empty_speakers_info.append([hearing_id, speaker])
    return empty_speakers_info


def fill_empty_speakers(csv_file, parsed_dir):
    with open(csv_file, 'r') as f:
        csvreader = csv.reader(f, delimiter='\t')
        next(csvreader, None)  # skip the headers
        for row in csvreader:
            hearing_id = row[0]
            speaker = row[1]
            speaker_full = row[2]
            speaker_type = row[3]

            parsed_file = os.path.join(parsed_dir, hearing_id + "_parsed.json")
            with open(parsed_file, 'r') as json_data:
                hearing = json.load(json_data)
            turns = hearing[constants.HEARING_TURNS]
            for turn in turns:
                if turn[constants.TURN_SPEAKER] == speaker:
                    turn[constants.TURN_SPEAKER_FULL] = speaker_full
                    turn[constants.TURN_SPEAKER_TYPE] = speaker_type
            with open(parsed_file, 'w') as f:
                json.dump(hearing, f)


def fix_misspellings(csv_file, parsed_dir):
    with open(csv_file, 'r') as f:
        csvreader = csv.reader(f, delimiter='\t')
        for row in csvreader:
            hearing_id = row[0]
            speaker_misspelled = row[1]
            speaker = row[2]
            speaker_full = row[3]
            speaker_type = row[4]

            parsed_file = os.path.join(parsed_dir, hearing_id + "_parsed.json")
            print("Reading parsed file: ", parsed_file)
            with open(parsed_file, 'r') as json_data:
                hearing = json.load(json_data)
            turns = hearing[constants.HEARING_TURNS]
            for turn in turns:
                if turn[constants.TURN_SPEAKER] == speaker_misspelled:
                    turn[constants.TURN_SPEAKER] = speaker
                    turn[constants.TURN_SPEAKER_FULL] = speaker_full
                    turn[constants.TURN_SPEAKER_TYPE] = speaker_type
            with open(parsed_file, 'w') as f:
                json.dump(hearing, f)


def check_duplicates(csv_file):
    with open(csv_file, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        next(csvreader, None)  # skip the headers

        prev_speaker = None
        for row in csvreader:
            curr_speaker = row[1]
            if curr_speaker == prev_speaker:
                print("Two consecutive turns with same speaker: ", row[0], curr_speaker)
            prev_speaker = curr_speaker


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fix Hearings 1.0')
    parser.add_argument('--option', type=str, help='Option to save or fix empty speakers', default='save')
    parser.add_argument('--parsed_dir', type=str, help='Path to parsed congressional hearings')
    parser.add_argument('--congress_num', type=int, help='Number of Congress to fix')
    parser.add_argument('--save_dir', type=str, help='save path')
    parser.add_argument('--fix_file', type=str, help='Tsv file with fixed speakers')
    args = parser.parse_args()
    print("Parser args: ", args)
    if args.option == 'save':
        save_empty_speakers(args.parsed_dir, args.congress_num, args.save_dir)
    elif args.option == 'fill':
        fill_empty_speakers(args.fix_file, args.parsed_dir)
    elif args.option == 'spell':
        fix_misspellings(args.fix_file, args.parsed_dir)
    elif args.option == 'duplicate':
        check_duplicates(args.fix_file)
