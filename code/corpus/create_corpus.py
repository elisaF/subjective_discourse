import argparse
from corpus import constants
import json
import os
import re
import spacy

from string import punctuation


def parse_congress(json_file, save_path, overwrite_files):
    print("Processing json file ", json_file, " and saving parsed hearings to ", save_path)
    num_parsed = 0
    num_existing = 0
    num_unavailable = 0
    num_no_witness = 0
    num_no_interview = 0
    num_empty_turns = 0

    with open(json_file, 'r') as json_data:
        hearings = json.load(json_data)
        for hearing in hearings:
            parse_result = parse_hearing(hearing, save_path, overwrite_files)
            if parse_result == "success":
                num_parsed += 1
            elif parse_result == "existing":
                num_existing += 1
            elif parse_result == "unavailable":
                num_unavailable += 1
            elif parse_result == "no_witness":
                num_no_witness += 1
            elif parse_result == "no_interview":
                num_no_interview += 1
            elif parse_result == "empty_turns":
                num_empty_turns += 1
    print("Parsed a total of ", len(hearings), " hearings. \n\tsuccessfully parsed: ", num_parsed,
          ", existing (skipped): ", num_existing, ", unavailable: ", num_unavailable,
          ", no interview: ", num_no_interview, ", no witness: ", num_no_witness, ", empty turns: ", num_empty_turns)


def parse_hearing(hearing_dict, save_path, overwrite_files):
    num_questions = 0
    hearing_id = hearing_dict[constants.HEARING_ID]
    hearing_text = hearing_dict[constants.HEARING_TEXT]

    json_file = os.path.join(save_path, hearing_id + '_parsed.json')
    titles_regex = '|'.join(constants.TITLES)

    print("Parsing congressional hearing ", hearing_id)

    # skip if file already exists
    if not overwrite_files:
        if os.path.isfile(json_file):
            print("Skipping because file already exists and overwrite option is False")
            return "existing"
    # first, check if the text is available
    if not is_text_available(hearing_text):
        print("Skipping because text is not available.", hearing_id)
        return "unavailable"
    # second, check if there are are witnesses in report
    witness_pattern = re.compile(r'\n.*(WITNESS|Witness|WITNESSES|WITNESSES:|Witnesses|Witnesses:)\s*\n')
    witness_matched = witness_pattern.search(hearing_text)
    if witness_matched is None:
        print("Skipping because there are no witnesses.")
        return "no_witness"

    # find end of hearing
    adjourned_pattern_no_brackets = re.compile(r'\n\s*.*(were|are|was|is|stands)\s?(adjourned|closed)\.\s*(Thank you.)?\n')
    match_adjourned = adjourned_pattern_no_brackets.search(hearing_text)
    if match_adjourned is None:
        adjourned_pattern = re.compile(
            r'\n\s*\[?.*(adjourned|concluded|(closed session)|(hearing ended))( at .*)?\.(\]|\))\s*\n')
        match_adjourned = adjourned_pattern.search(hearing_text)
        if match_adjourned is None:
            no_hearings_pattern = re.compile(r'\n\s*\[.*unable to hold')
            match_no_hearings = no_hearings_pattern.search(hearing_text)
            if match_no_hearings:
                print("Skipping because there are no in-person interviews.")
                return "no_interview"
            else:
                # print(hearing_text)
                # raise ValueError("Couldn't find end of hearing!", hearing_id)
                last_line_pattern = re.compile(r'</pre></body></html>')
                match_adjourned = last_line_pattern.search(hearing_text)
                print("WARNING: Couldn't find end of hearing, so using last line!", hearing_id)
    hearing_text = (hearing_text[:match_adjourned.end()])

    # find beginning of hearing
    statement_pattern = re.compile(r'\n\s*\[.*(opening |prepared )?(statement|testimony) of .*\]\s*\n.*\n.*\n',
                                   re.IGNORECASE)
    beginning_matches = list(statement_pattern.finditer(hearing_text))

    # also check for statements without brackets
    statement_pattern_no_brackets = re.compile(r'\n\s*.*(Opening |OPENING |Prepared | PREPARED)?(Statement|STATEMENT|Testimony|TESTIMONY) (of|OF) [A-Z].*\s*\n.*\n.*\n')
    beginning_matches_no_brackets = list(statement_pattern_no_brackets.finditer(hearing_text))
    if not beginning_matches:
        beginning_matches = beginning_matches_no_brackets
    elif beginning_matches_no_brackets:
        if beginning_matches_no_brackets[-1].end() > beginning_matches[-1].end():
            beginning_matches = beginning_matches_no_brackets
    if not beginning_matches:
        witness_present_pattern = re.compile(r'\n\s*witness(es)? present:\s*.*\n', re.IGNORECASE)
        beginning_matches = list(witness_present_pattern.finditer(hearing_text))
        if not beginning_matches:
            # if there are no statements
            begin_hearing_pattern = re.compile(r'\n.*hearing to order\.\s+.*\n'.format(titles_regex))
            beginning_matches = list(begin_hearing_pattern.finditer(hearing_text))
            if not beginning_matches:
                # look for attachments of statements
                begin_hearing_pattern = re.compile(r'\n\s* \[The information follows:\]\s*\n')
                beginning_matches = list(begin_hearing_pattern.finditer(hearing_text))
                if not beginning_matches:
                    # print(hearing_text)
                    # raise ValueError("Couldn't find start of hearing!", hearing_id)
                    print("WARNING: Couldn't find start of hearing, so skipping!", hearing_id)
                    return "no_interview"
    hearing_text_beginning = hearing_text[:beginning_matches[-1].end()]
    hearing_text = hearing_text[beginning_matches[-1].end():]
    # split into turns
    pattern = re.compile(r'\n\s+(((?:{})\s+[A-Z]+[a-zA-Z\-]+ ?[A-Z\[]?[a-zA-Z\-\]]*\.)|(The Chairman\.))\s+\w+'.format(titles_regex))
    matches = pattern.finditer(hearing_text)
    turn_starts = []
    if matches:
        for match in matches:
            turn_starts.append(match.start())
    turn_texts = []
    for index in range(0, len(turn_starts)):
        if index == len(turn_starts) - 1:
            end_turn = -1
        else:
            end_turn = turn_starts[index + 1]
        turn_text = hearing_text[turn_starts[index]:end_turn]
        turn_texts.append(turn_text)
    # remove new line
    # turn_texts = [turn_text.replace("\n", "") for turn_text in turn_texts]
    # pattern_speaker = re.compile(r'^ *(?:{}) [A-Z]+[a-zA-Z\-]+ ?[A-Z\[]?[a-zA-Z\-\]]*\. '.format(titles_regex))
    pattern_speaker = re.compile(r'\n\s+(((?:{})\s+[A-Z]+[a-zA-Z\-]+ ?[A-Z\[]?[a-zA-Z\-\]]*\.)|(The Chairman\.))\s+'.format(titles_regex))

    turn_dicts = []
    is_answer = False
    for turn_idx, turn_text in enumerate(turn_texts):
        turn_dict = {constants.TURN_ID: turn_idx}
        speaker_match = pattern_speaker.match(turn_text)
        if speaker_match is None:
            raise ValueError("Couldn't find speaker in hearing ", hearing_id)
        speaker = turn_text[speaker_match.start():speaker_match.end()].strip(' .')
        # strip off any comments like when speaker is interrupted
        if "[" in speaker:
            speaker = speaker[:speaker.find("[")].rstrip()
        turn_dict[constants.TURN_SPEAKER] = speaker.strip().replace("\n", "")

        text = turn_text[speaker_match.end():].strip()
        text = text.replace("\n", "")
        if len(text) == 0:
            if len(turn_dicts) > 0:
                print(turn_dicts[-1])
            print("Text is empty for speaker, ", speaker, "! Skipping this turn")
        else:
            turn_dict['text'] = text
            has_question = has_ending_question(text)
            turn_dict['is_question'] = has_question
            turn_dict['is_answer'] = is_answer
            turn_dicts.append(turn_dict)
            if has_question:
                is_answer = True
                num_questions += 1
            else:
                is_answer = False

    if len(turn_dicts) == 0:
        print("WARNING: Parsing skipped all turns. Nothing left of hearing!")
        return "empty_turns"

    # get info on speakers
    speaker_to_desc = {}
    witness_matched = witness_pattern.search(hearing_text_beginning)
    if witness_matched:
        hearing_text_witness = hearing_text_beginning[witness_matched.start():]
        hearing_text_politician = hearing_text_beginning[:witness_matched.start()]
        # skip title section, if present
        if 'for sale by' in hearing_text_politician.lower():
            hearing_text_politician = hearing_text_politician[hearing_text_politician.lower().index('for sale by'):]
        for turn_dict in turn_dicts:
            speaker_desc = None
            speaker_type = None  # can be witness or politician
            speaker = turn_dict[constants.TURN_SPEAKER]
            if speaker not in speaker_to_desc:
                speaker_name = ' '.join(speaker.split(' ')[1:]).strip().strip(punctuation)  # assume first part is title
                pattern_witness_name = re.compile(r'\n.*{speaker}((.|\n)*?)(\.){{2,}}\s+\d+\n'.format(speaker=speaker_name),
                                                  flags=re.IGNORECASE)
                witness_match = pattern_witness_name.search(hearing_text_witness)

                pattern_politician_name = re.compile(r'.*{speaker}(.)*?((\s){{2,}}|\n)'.format(speaker=speaker_name),
                                                     flags=re.IGNORECASE)
                politician_match = pattern_politician_name.search(hearing_text_politician)

                if politician_match:
                    speaker_desc = hearing_text_politician[politician_match.start():politician_match.end()].replace('\n',
                                                                                                                    '').strip()
                    # fix case where two politician names are on the same line and we want second one
                    splits = speaker_desc.split('  ')
                    if len(splits) > 1:
                        speaker_desc = [split for split in splits if speaker_name.lower() in split.lower()][0].strip()
                    speaker_type = constants.TURN_SPEAKER_TYPE_POLITICIAN
                elif witness_match:
                    speaker_line = hearing_text_witness[witness_match.start():witness_match.end()]
                    speaker_desc = ' '.join([line.strip() for line in speaker_line[:speaker_line.index('..')].split('\n')])
                    if 'oral statement' in speaker_desc.lower():
                        speaker_desc = speaker_desc[:speaker_desc.lower().index('oral statement')]
                    speaker_type = constants.TURN_SPEAKER_TYPE_WITNESS
                else:
                    print("WARNING: No match found to get full description for speaker", speaker, "!")
                    speaker_to_desc[speaker] = speaker_desc
            else:
                speaker_desc = speaker_to_desc[speaker]
            turn_dict[constants.TURN_SPEAKER_FULL] = speaker_desc
            turn_dict[constants.TURN_SPEAKER_TYPE] = speaker_type
    else:
        print("WARNING: Witness section not found before all of the statements, so skipping!")
        return "no_witness"

    parsed_hearing_dict = {constants.HEARING_ID: hearing_dict[constants.HEARING_ID],
                           constants.HEARING_TITLE: hearing_dict[constants.HEARING_TITLE],
                           constants.HEARING_DATES: hearing_dict[constants.HEARING_DATES],
                           constants.HEARING_CONGRESS: hearing_dict[constants.HEARING_CONGRESS],
                           constants.HEARING_TURNS: turn_dicts}

    with open(json_file, 'w') as f:
        json.dump(parsed_hearing_dict, f)
    print("Saved congressional hearing ", hearing_id, " with ", len(turn_dicts), " turns and ", num_questions, " questions.")
    return "success"


def is_text_available(text):
    # text is not available at all (only pdf)
    if "\"status\":404,\"error\":\"Not Found\",\"message\":\"No message available\"" in text:
        return False
    # text is for sale
    elif "For sale by the U.S. Government Printing Office" in text:
        return False
    else:
        return True


def has_ending_question(text, max_num_ending_sents=3):
    nlp = spacy.load("en")
    doc = nlp(text)
    sents = [sent.text for sent in list(doc.sents)]
    if len(sents) > max_num_ending_sents:
        sents = sents[max_num_ending_sents*-1:]
    pattern_question = re.compile(r'.*\?( |$)')
    for sent in sents:
        question_match = pattern_question.match(sent)
        if question_match:
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Corpus Creation 1.0')
    parser.add_argument('--congress_file', type=str, help='Congress json file')
    parser.add_argument('--save_path', type=str, help='save path')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing parsed hearings')
    args = parser.parse_args()
    print("Parser args: ", args)
    parse_congress(args.congress_file, args.save_path, args.overwrite)
