import argparse
from corpus import constants
import json
import os
import requests
import time

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


def get_hearings_json(first_congress, last_congress, save_dir):
    print("Going to save hearings from Congress #", first_congress, "to #", last_congress, "to directory", save_dir)
    hearings_counter = 0
    s = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=0.1,
                    status_forcelist=[500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retries))

    headers = {
        'accept': 'application/json',
    }

    # loop over every congress
    for congress_num in range(first_congress, last_congress+1):
        raw_hearings = []
        start = time.time()
        params = (
            ('offset', '0'),
            ('pageSize', '10000'),
            ('congress', congress_num),
            ('api_key', constants.API_KEY),
        )

        packages_response = s.get('https://api.govinfo.gov/collections/CHRG/2018-01-01T00%3A00%3A00Z', headers=headers,
                            params=params)
        hearings = packages_response.json()[constants.JSON_PACKAGES]

        # loop over every hearing
        for hearing in hearings:
            hearings_counter += 1
            hearing_dict = {}
            hearing_id = hearing[constants.JSON_ID]
            hearing_title = hearing[constants.JSON_TITLE]
            print("\t", hearing_id, hearing_title)

            # download htm (text) file for hearing
            h_headers = {
                'accept': '*/*',
            }

            h_params = (
                ('api_key', constants.API_KEY),
            )

            summary_request_response = s.get('https://api.govinfo.gov/packages/' + hearing_id + '/summary',
                                                    headers=h_headers,
                                                    params=h_params)

            summary_json = summary_request_response.json()
            if constants.JSON_DATES in summary_json:
                hearing_dates = summary_json[constants.JSON_DATES]
            else:
                hearing_dates = summary_json[constants.JSON_DATES2]

            hearing_request_response = s.get('https://api.govinfo.gov/packages/'+hearing_id+'/htm', headers=h_headers,
                                params=h_params)

            hearing_text = hearing_request_response.text

            hearing_dict[constants.HEARING_ID] = hearing_id
            hearing_dict[constants.HEARING_TITLE] = hearing_title
            hearing_dict[constants.HEARING_DATES] = hearing_dates
            hearing_dict[constants.HEARING_TEXT] = hearing_text
            hearing_dict[constants.HEARING_CONGRESS] = congress_num
            raw_hearings.append(hearing_dict)

        with open(os.path.join(save_dir, 'congressional_hearings_'+str(congress_num)+'.json'), 'w') as f_out:
            json.dump(raw_hearings, f_out)

        end = time.time()
        print("Time elapsed:", str((end-start)/60), "minutes.\nFinished processing", str(len(raw_hearings)),
              "congressional hearings for the Congress #", str(congress_num))

    print("Saved ", str(hearings_counter), " congressional hearings to directory ", save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Download Data 1.0')
    parser.add_argument('--first_congress', type=int, help='Number of first congress to download')
    parser.add_argument('--last_congress', type=int, help='Number of last congress to download')
    parser.add_argument('--save_path', type=str, help='save path')
    args = parser.parse_args()
    print("Parser args: ", args)
    get_hearings_json(args.first_congress, args.last_congress, args.save_path)
