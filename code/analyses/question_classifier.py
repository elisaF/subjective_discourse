# Sources:
# https://github.com/OlaSearch/qaclassifer/blob/master/qaclassifier/classify.py
# https://towardsdatascience.com/linguistic-rule-writing-for-nlp-ml-64d9af824ee8#762b

import stanza

stanza.download('en')

WH_TAGS = ['WDT', 'WP', 'WP$', 'WRB']


class Sentence(object):
    def __init__(self, sentence):
        self.sentence_ = sentence
        self.root_ = None

    def root(self):
        if not self.root_:
            self.root_ = [word for word in self.sentence_.words if word.deprel == 'root'][0]
        return self.root_


class Word(object):
    def __init__(self, word, sentence):
        self.word_ = word
        self.sentence_ = sentence
        self.head_ = None
        self.children_ = []
        self.lefts_ = []
        self.rights_ = []

    def children(self):
        if not self.children_:
            parent_id = int(self.word_.id)
            for w in self.sentence_.words:
                if w.head == parent_id:
                    self.children_.append(w)
        return self.children_

    def lefts(self):
        if not self.lefts_:
            children = self.children()
            for child in children:
                if int(child.id) < int(self.word_.id):
                    self.lefts_.append(child)
        return self.lefts_

    def rights(self):
        if not self.rights_:
            children = self.children()
            for child in children:
                if int(child.id) > int(self.word_.id):
                    self.rights_.append(child)
        return self.rights_

    def head(self):
        if not self.head_:
            parent_id = self.word_.head
            for word in self.sentence_.words:
                if parent_id == int(word.id):
                    self.head_ = word
                    break
        return self.head_


def is_wh_question(sentence):
    wh_words = [t for t in sentence.words if t.xpos in WH_TAGS and t.feats and 'PronType=Int' in t.feats.split('|')]
    if wh_words:
        return True
    return False


def is_tag_question(sentence):
    commas = [t for t in sentence.words if t.xpos == ","]
    if commas:
        id_after_comma = str(int(commas[-1].id) + 1)
        word_after_comma = [t for t in sentence.words if t.id == id_after_comma][0]
        if word_after_comma.xpos == "MD" or word_after_comma.xpos.startswith("VB"):
            if sentence.words[-2].xpos == "PRP" or sentence.words[-3].xpos == "PRP":
                return True
    return False


def is_declarative_question(sentence):
    return not is_wh_question(sentence)


def _is_subject(word):
    subject_deps = {"csubj", "nsubj", "nsubjpass"}
    return word.deprel in subject_deps


def is_polar_question(sentence):
    if is_wh_question(sentence):
        return False

    root = Word(Sentence(sentence).root(), sentence)
    root_children = root.children()
    subj = [w for w in root_children if _is_subject(w)]

    # Type I: In a non-copular sentence, "is" is an aux.
    # "Is she using spaCy?" or "Can you read that article?"
    aux = [t for t in root.lefts() if t.deprel == "aux"]
    if subj and aux:
        return int(aux[0].id) < int(subj[0].id)

    # Type II: In a copular sentence, "is" is the main verb.
    # "Is the mouse dead?"
    verb = [t for t in root.lefts() if t.xpos.startswith('V')]
    if subj and verb:
        return int(verb[0].id) < int(subj[0].id)

    return False


def is_or_question(sentence):
    for word in sentence.words:
        if word.deprel == "cc" and word.text == "or":
            return True
    return False


def get_last_question(doc):
    for sentence in reversed(doc.sentences):
        if sentence.words[-1].text == "?":
            return sentence


def get_question_types(response_tuples):
    nlp = stanza.Pipeline('en', processors={'tokenize': 'spacy'})
    types = []
    for response_tuple in response_tuples:
        types.append(get_question_type(response_tuple, nlp))
    return types


def get_question_type(response_tuple, nlp):
    id = response_tuple[0]
    response = response_tuple[1]
    doc = nlp(response)
    # analyze only last question
    sentence = get_last_question(doc)
    # reparse last question after removing whitespace
    sentence = nlp(sentence.text.strip(' ')).sentences[0]
    if is_or_question(sentence):
        return tuple((id, "OR"))
    elif is_wh_question(sentence):
        return tuple((id, "WH"))
    elif is_polar_question(sentence):
        return tuple((id, "YN"))
    elif is_tag_question(sentence):
        return tuple((id, "TG"))
    elif is_declarative_question(sentence):
        return tuple((id, "DC"))
    else:
        return tuple((id, "other"))


if __name__ == '__main__':
    sents = [tuple((0, "Is that for real?")),
             tuple((1, "Can you stop?")),
             tuple((2, "Do you love John?")),
             tuple((3, "Are you sad?")),
             tuple((4, "Was she singing?")),
             tuple((5, "Won't you come over for dinner?")),
             tuple((6, "Would you help me?")),
             tuple((7, "Who do you love?")),
             tuple((8, "Whose child is that?")),
             tuple((9, "How do you know him?")),
             tuple((10, "To whom did she read the book?")),
             tuple((11, "I'm hungry?")),
             tuple((12, "Tell me what you mean?")),  # fails
             tuple((13, "Would love to help you?")),
             tuple((14, "Don't be sad?")),
             tuple((15, "Whatever you want?")),  # fails
             tuple((16, "What you say is impossible?")),  # fails
             tuple((17, "Where you go, I will go?")),  # fails
             tuple((18, "Papua, New Guinea?")),
             tuple((19, "Thank you.    Do you recall this document?")),
             tuple((20, "Thank you. Do you recall this document?"))]
    q_types = get_question_types(sents)
    print(list(zip(q_types, sents)))
