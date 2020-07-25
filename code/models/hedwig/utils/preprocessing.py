import re
import torch


def clean_string(string):
    """
    Performs tokenization and string cleaning for the Reuters dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()


def split_sents(string):
    string = re.sub(r"[!?]"," ", string)
    return string.strip().split('.')


def generate_ngrams(tokens, n=2):
    n_grams = zip(*[tokens[i:] for i in range(n)])
    tokens.extend(['-'.join(x) for x in n_grams])
    return tokens


def process_labels(string):
    """
    Returns the label string as a list of integers
    """
    return [float(x) for x in string]


def pad_input_matrix(unpadded_matrix, max_doc_length):
    """
    Returns a zero-padded matrix for a given jagged list
    :param unpadded_matrix: jagged list to be padded
    :return: zero-padded matrix
    """
    max_doc_length = min(max_doc_length, max(len(x) for x in unpadded_matrix))
    zero_padding_array = [0 for i0 in range(len(unpadded_matrix[0][0]))]

    for i0 in range(len(unpadded_matrix)):
        if len(unpadded_matrix[i0]) < max_doc_length:
            unpadded_matrix[i0] += [zero_padding_array for i1 in range(max_doc_length - len(unpadded_matrix[i0]))]
        elif len(unpadded_matrix[i0]) > max_doc_length:
            unpadded_matrix[i0] = unpadded_matrix[i0][:max_doc_length]


def get_coarse_labels(label_ids, num_coarse_labels, parent_to_child_index_map, device):
    coarse_label_ids = torch.empty(label_ids.shape[0], num_coarse_labels, dtype=torch.long, device=device)
    for parent_idx, child_idxs in parent_to_child_index_map.items():
        child_labels = torch.index_select(label_ids, 1, torch.tensor(child_idxs, dtype=torch.long, device=device))
        coarse_label_ids[:,parent_idx] = child_labels.byte().any(dim=1)
    return coarse_label_ids


def get_fine_mask(gold_coarse_labels, parent_to_child_index_map):
    masks = []
    for parent_idx, child_idxs in parent_to_child_index_map.items():
        masks.append(gold_coarse_labels[:, parent_idx].repeat(len(child_idxs), 1).transpose(0, 1).bool())
    mask = torch.cat(masks, 1)
    return mask
