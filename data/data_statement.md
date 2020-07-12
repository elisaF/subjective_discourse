# Data Statement for SubjectiveResponses

Data set name: SubjectiveResponses

Citation (if available): TBD

Data set developer(s): Elisa Ferracane

Data statement author(s): Elisa Ferracane

Others who contributed to this document: N/A

## A. CURATION RATIONALE 

The purpose of this dataset is to capture subjective judgments of responses to questions. We choose witness testimonials in U.S. congressional hearings because they contain question-answer sessions, are often controversial and elicit subjectivity from untrained crowdsourced workers. The data is sourced from publicly available transcripts provided by the U.S. government (https://www.govinfo.gov/app/collection/chrg) and downloaded using their provided APIs (https://api.govinfo.gov/docs/). We download all transcripts from 113th-116th congresses (available as of September 18, 2019), then use regexes to identify speakers, turns, and turns containing questions. We retain hearings with only one witness and with more than 100 question-response pairs as a signal of argumentativeness. To ensure a variety of topics and political leanings, we sample hearings from each congress and eliminate those whose topic is too unfamiliar to an average American citizen (e.g. discussing a task force in the Nuclear Regulatory Commission). This process yields a total of 20 hearings: 4 hearings from the 113th congress (CHRG-113hhrg86195, CHRG-113hhrg88494, CHRG-113hhrg89598 CHRG-113hhrg93834), 5 hearings from the 114th (CHRG-114hhrg20722, CHRG-114hhrg22125, CHRG-114hhrg26003, CHRG-114hhrg95063, CHRG-114hhrg97630), 7 hearings from the 115th (CHRG-115hhrg25545, CHRG-115hhrg30242, CHRG-115hhrg30956, CHRG-115hhrg31349, CHRG-115hhrg31417, CHRG-115hhrg31504,  CHRG-115hhrg32380), and 4 hearings from the 116th (CHRG-116hhrg35230, CHRG-116hhrg35589, CHRG-116hhrg36001, CHRG-116hhrg37282). For annotation, we then select the first 50 question-response pairs from each hearing.

Code used to create the dataset is available at http://github.com/anonymous.

## B. LANGUAGE VARIETY/VARIETIES

* BCP-47 language tag: en-US
* Language variety description: American English as spoken in U.S. governmental setting

## C. SPEAKER DEMOGRAPHIC

* Description: The speakers are from two groups: the questioners are politicians (members of Congress) and the witnesses can be politicians, businesspeople or other members of the general public. 
* Age: No specific information was collected about the ages, but all are presumed to be adults (30+ years old).
* Gender: No specific information was collected about gender, but members of Congress include both men and women. The witnesses included both men and women.
* Race/ethnicity (according to locally appropriate categories): No information was collected.
* First language(s): No information was collected.
* Socioeconomic status: No information was collected.
* Number of different speakers represented: 91 members of Congress and 20 witnesses.
* Presence of disordered speech: No information was collected but none is expected.
 
## D. ANNOTATOR DEMOGRAPHIC

Annotators:
* Description: Workers on the Amazon Mechanical Turk platform who reported to live in the U.S. and had a >95% approval rating with >500 approved HITs were recruited during the time period of November 2019 - March 2020.
* Age: No information was collected.
* Gender: No information was collected.
* Race/ethnicity (according to locally appropriate categories): No information was collected.
* First language(s): No information was collected.
* Training in linguistics/other relevant discipline: None.

Annotation guideline developer:
* Description: The author of this data statement.
* Age: 40.
* Gender: Female.
* Race/ethnicity (according to locally appropriate categories): Hispanic.
* First language(s): American English.
* Training in linguistics/other relevant discipline: PhD candidate in computational linguistics.


## E. SPEECH SITUATION

* Description: Witness testimonials in U.S. congressional hearings spanning the 114th-116th Congresses.
* Time: 2013-2019
* Place: U.S. Congress
* Modality (spoken/signed, written): transcribed from spoken.
* Scripted/edited vs. spontaneous: mostly spontaneous, though members of Congress sometimes read questions they have written down
* Synchronous vs. asynchronous interaction: synchronous
* Intended audience:  the U.S. government and the general public, as all hearings are both transcribed and televised

## F. TEXT CHARACTERISTICS

The genre is political discourse in a highly structured setting where a chairperson runs the meeting, and each member of Congress is afforded 5 minutes to question the witness but can yield their time to others. Topics vary based on the congressional committee that is holding the hearing, and include oversight of other governmental bodies (e.g., IRS, Department of Justice) and inquiries into businesses suspected of misconduct (e.g., FaceBook, Wells Fargo).

## G. RECORDING QUALITY

N/A

## H. OTHER

N/A

## I. PROVENANCE APPENDIX

N/A

## About this document

A data statement is a characterization of a dataset that provides context to allow developers and users to better understand how experimental results might generalize, how software might be appropriately deployed, and what biases might be reflected in systems built on the software.

Data Statements are from the University of Washington. Contact: [datastatements@uw.edu](mailto:datastatements@uw.edu). This document template is licensed as [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/).

This version of the markdown Data Statement is from June 4th 2020. The Data Statement template is based on worksheets distributed at the [2020 LREC workshop on Data Statements](https://sites.google.com/uw.edu/data-statements-for-nlp/), by Emily M. Bender, Batya Friedman, and Angelina McMillan-Major. Adapted to community Markdown template by Leon Dercyznski.