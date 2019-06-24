import torch
import re
import numpy as np


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def LCS(x, y):
    """ Longest Common Sequence"""
    matrix = [''] * (len(x) + 1)
    for index_x in range(len(matrix)):
        matrix[index_x] = [''] * (len(y) + 1)

    for index_x in range(1, len(x) + 1):
        for index_y in range(1, len(y) + 1):
            # end with same char -> lcs = substring lcs + ending char
            if x[index_x - 1] == y[index_y - 1]:
                matrix[index_x][index_y] = matrix[index_x -
                    1][index_y - 1] + x[index_x - 1]

            # ending with distinct char, consider 2 cases: 1 - substr of x and y ; 2 - substr of y and x
            elif len(matrix[index_x][index_y - 1]) > len(matrix[index_x - 1][index_y]):
                matrix[index_x][index_y] = matrix[index_x][index_y - 1]
            else:
                matrix[index_x][index_y] = matrix[index_x - 1][index_y]

    return matrix[len(x)][len(y)]


def rouge(hyp, ref, n):
    scores = []
    for h, r in zip(hyp, ref):
        r = re.sub(r'[UNK]', '', r)
        r = re.sub(r'[’!"#$%&\'()*+,-./:：？！《》;<=>?@[\\]^_`{|}~]+', '', r)
        r = re.sub(r'\d', '', r)
        r = re.sub(r'[a-zA-Z]', '', r)
        count = 0
        match = 0
        for i in range(len(r) - n):
            gram = r[i:i + n]
            if gram in h:
                match += 1
            count += 1
        scores.append(match / count)
    return np.average(scores)


def rougeL(hyp, ref):
    """ calc rogue L metric score
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf
    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / \
                (R_lcs + (beta^2) * P_lcs)
    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary
    Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries
    Returns:
    A list of floats: scores
    """


    scores = []
    # print(list(zip(ref,hyp)))
    for h, r in zip(hyp, ref):
        r = re.sub(r'[UNK]', '', r)
        r = re.sub(r'[’!"#$%&\'()*+,-./:：？！《》;<=>?@[\\]^_`{|}~]+', '', r)
        r = re.sub(r'\d', '', r)
        r = re.sub(r'[a-zA-Z]', '', r)

        #lcs
        lcs = len(LCS(h,r))
        m = len(h)
        n = len(r)
        #score
        r_lcs = lcs / m
        p_lcs = lcs / n
        beta = p_lcs / (r_lcs + 1e-12)
        num = (1 + (beta**2)) * r_lcs * p_lcs
        denom = r_lcs + ((beta**2) * p_lcs)
        scores.append(num / (denom + 1e-12))


    return np.average(scores)


if __name__ == "__main__":
    hyp = ['交大闵行校区一实验室发生硫化氢泄漏事故中无学生伤亡', '[UNK]史上最严的环保法[UNK]']
    ref = ['上海交大闵行校区：实验室换瓶时硫化氢泄漏送货员身亡', '#2015全国两会#傅莹：[UNK]史上最严[UNK]环保法是[UNK]有牙齿[UNK]的']
    print(rouge(hyp, ref, 2))
    print(rouge(hyp,ref, 3))
    print(rougeL(hyp,ref))

        
