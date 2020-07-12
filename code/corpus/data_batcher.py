from corpus import constants, batch_constants as bc, summaries
from common import utils

import argparse
import csv
import json
import numpy as np
import os
import pandas as pd


def prepare_hearings(hearing_files, save_dir):
    for hearing_file in hearing_files:
        write_hearing_to_csv(hearing_file, save_dir)


def write_hearing_to_csv(hearing_file, save_dir):
    print("Writing hearing to csv: ", hearing_file)
    rows = []
    with open(hearing_file, 'r') as json_data:
        hearing = json.load(json_data)
    hearing_id = hearing[constants.HEARING_ID]
    title = hearing[constants.HEARING_TITLE]
    dates = hearing[constants.HEARING_DATES][0]
    turns = hearing[constants.HEARING_TURNS]
    rows.append([hearing_id, title, dates, None, None, None])
    for idx, turn in enumerate(turns):
        speaker = turn[constants.TURN_SPEAKER]
        speaker_full = turn[constants.TURN_SPEAKER_FULL]
        speaker_type = turn[constants.TURN_SPEAKER_TYPE]
        is_question = turn[constants.TURN_IS_QUESTION]
        is_answer = turn[constants.TURN_IS_ANSWER]
        text = turn[constants.HEARING_TEXT]
        rows.append([idx, speaker, speaker_full, speaker_type, is_question, is_answer, text])
    save_file = os.path.join(save_dir, hearing_id + '.csv')
    with open(save_file, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerows(rows)


def pair(csv_files):
    for csv_file in csv_files:
        print("Processing csv file ", csv_file)
        with open(csv_file, 'r') as f:
            csvreader = csv.reader(f, delimiter=',')
            headers = next(csvreader)
            csvreader = list(csvreader)
        id = headers[0]
        title = headers[1]
        date = headers[2]
        summary = summaries.summaries_dict[id]
        hearing_info = [id, title, date, summary]
        question_response_list = create_question_response_list(csvreader)
        question_response_list = [hearing_info + question_response + [False] * 3 for question_response in
                                  question_response_list]

        save_file = os.path.splitext(csv_file)[0] + "_qa.csv"
        print("Saving to file ", save_file)
        with open(save_file, 'w') as f:
            csvwriter = csv.writer(f, delimiter=',')
            csvwriter.writerow(bc.PAIRED_QA_HEADER)
            csvwriter.writerows(question_response_list)


def create_batch(csv_files, save_batch_file, batch_size=25, hit_size=5, keep_batched=False):
    list_hits = []
    for csv_file in csv_files:
        print("Processing csv file ", csv_file)
        df_qa_pairs = pd.read_csv(csv_file)

        # drop pairs that have already been batched
        if keep_batched:
            df_qa_pairs_unbatched = df_qa_pairs
        else:
            df_qa_pairs_unbatched = df_qa_pairs.loc[df_qa_pairs[bc.BATCHED]==False]

        # get the hearing info, then drop
        hearing_columns = [bc.BATCH_HEARING_ID, bc.BATCH_HEARING_TITLE, bc.BATCH_HEARING_DATE, bc.BATCH_HEARING_SUMMARY]
        hearing_info = df_qa_pairs_unbatched.iloc[0][hearing_columns].values.tolist()
        drop_columns = hearing_columns + [bc.BATCHED, bc.LABELED_TYPE, bc.LABELED_INTENT]
        df_qa_pairs_unbatched.drop(columns=drop_columns, inplace=True)
        # group into hits
        hits = list(map(list, utils.grouper_without_fill(df_qa_pairs_unbatched.values.tolist(), hit_size)))
        hits = utils.flatten(hits)

        # add hearing info to beginning of hit
        hits = [hearing_info + hit for hit in hits]
        list_hits.append(hits)

    # mix across hearings
    mixed_hits = list(utils.roundrobin(*list_hits))
    mixed_hits = mixed_hits[:batch_size]
    mixed_hits_df = pd.DataFrame(mixed_hits, columns=bc.BATCH_HEADER)
    mixed_hits_df.to_csv(save_batch_file, index=False)

    # update original csv file to mark qa pairs as batched
    for csv_file in csv_files:
        print("Marking qa pairs as batched for file ", csv_file)
        df_qa_pairs = pd.read_csv(csv_file)
        hearing_id = df_qa_pairs[bc.BATCH_HEARING_ID].values[0]
        turn_ids = mixed_hits_df.loc[mixed_hits_df[bc.BATCH_HEARING_ID]==hearing_id,
                                     [bc.BATCH_Q1_ID,bc.BATCH_Q2_ID, bc.BATCH_Q3_ID, bc.BATCH_Q4_ID, bc.BATCH_Q5_ID]].values.flatten()
        df_qa_pairs.loc[df_qa_pairs[bc.BATCH_Q1_ID].isin(turn_ids), bc.BATCHED] = True
        df_qa_pairs.to_csv(csv_file, index=False)


def create_question_response_list(csvreader):
    question_response_list = []

    # initialize values
    prev_id = -1
    prev_speaker = ''
    prev_speaker_type = ''
    prev_speaker_full = ''
    prev_text = ''

    next_id = np.inf
    next_speaker = ''
    next_speaker_type = ''
    next_speaker_full = ''
    next_text = ''

    for idx, curr_row in enumerate(csvreader):
        curr_id = curr_row[0]
        curr_speaker = curr_row[1]
        curr_speaker_full = curr_row[2]
        curr_speaker_type = curr_row[3]
        curr_text = curr_row[6]

        is_politician_question = True if curr_row[4].lower() == "true" \
                                         and curr_speaker_type == constants.TURN_SPEAKER_TYPE_POLITICIAN \
            else False
        # only save if we have a response, and it is from a witness
        if is_politician_question and (idx + 1) < len(csvreader) \
                and csvreader[idx + 1][3] == constants.TURN_SPEAKER_TYPE_WITNESS:
            response_row = csvreader[idx + 1]
            r_id = response_row[0]
            r_speaker = response_row[1]
            r_speaker_full = response_row[2]
            r_speaker_type = response_row[3]
            r_text = response_row[6]

            if (idx + 2) < len(csvreader):
                next_row = csvreader[idx + 2]
                next_id = next_row[0]
                next_speaker = next_row[1]
                next_speaker_full = next_row[2]
                next_speaker_type = next_row[3]
                next_text = next_row[6]
            question_response_list.append(
                [curr_id, curr_speaker, curr_speaker_type + ": " + curr_speaker_full, curr_text,
                 r_id, r_speaker, r_speaker_type + ": " + r_speaker_full, r_text,
                 prev_id, prev_speaker, prev_speaker_type + ": " + prev_speaker_full, prev_text,
                 next_id, next_speaker, next_speaker_type + ": " + next_speaker_full, next_text])
        prev_id = curr_id
        prev_speaker = curr_speaker
        prev_speaker_full = curr_speaker_full
        prev_speaker_type = curr_speaker_type
        prev_text = curr_text
    return question_response_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Annotation Batcher 1.0')
    parser.add_argument('--option', type=str, help='Option to prepare or create batches', default='prepare')

    parser.add_argument('--save_dir', type=str, help='Directory to save prepared annotation data (only valid if option is prepare)')

    parser.add_argument('--hearing_files', type=str,
                        help='Full path of comma-separated congressional files to include in batch')
    parser.add_argument('--save_batch_file', type=str, help='File to save batch hits (only valid if option is create)')
    parser.add_argument('--batch_size', type=int, help='Size of batch')
    parser.add_argument('-k', '--keep_batched', action='store_true', help="include all HITs, regardless of whether "
                                                                          "they've been batched before")

    args = parser.parse_args()
    print("Parser args: ", args)
    hearing_files = [file.strip() for file in args.hearing_files.split(',')]
    if args.option == 'prepare':
        prepare_hearings(hearing_files, args.save_dir)
    elif args.option == 'pair':
        pair(hearing_files)
    elif args.option == 'create':
        create_batch(hearing_files, args.save_batch_file, batch_size=args.batch_size, keep_batched=args.keep_batched)
