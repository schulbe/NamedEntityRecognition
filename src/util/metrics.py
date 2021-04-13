    def get_precision_recall(ner, labels, max_num, seed_docs):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    false_pos_list = list()

    for ix in [i for i in range(max_num) if i not in seed_docs]:
        probas = ner.predict_token_probabilities_iterated(ner.corpus[ix].strip(), thres=0.499, mask_thres=0.5)
        for token, prob in probas:
            if ix in labels:
                if prob > 0 and token in labels[ix]:
                    true_pos += 1
                elif prob > 0 and token not in labels[ix]:
                    false_pos += 1
                    false_pos_list.append((ix, token))
                elif prob == 0 and token in labels[ix]:
                    false_neg += 1
                else:
                    true_neg += 1
            else:
                if prob > 0:
                    false_pos += 1
                    false_pos_list.append((ix, token))
                else:
                    true_neg += 1

    try:
        precision = true_pos/(true_pos+false_pos)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = true_pos/(true_pos+false_neg)
    except ZeroDivisionError:
        recall = 0

    try:
        f_score = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        f_score=0

    support = true_pos+false_neg
    return precision, recall, f_score, support
    # return (precision, recall, f_score, support), false_pos_list


def get_precision_recall_spacy(nlp, labels, max_num, corpus, tokenize_fn, seed_docs):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for ix in [i for i in range(max_num) if i not in seed_docs]:
        doc = nlp(corpus[ix])
        names = list()
        for ent in doc.ents:
            if ent.label_ == 'PER':
                names.extend(tokenize_fn(ent.text))

        for token in tokenize_fn(corpus[ix]):
            if ix in labels:
                if token in names and token.lower() in labels[ix]:
                    true_pos += 1
                elif token in names and token.lower() not in labels[ix]:
                    false_pos += 1
                elif token not in names and token.lower() in labels[ix]:
                    false_neg += 1
                else:
                    true_neg += 1
            else:
                if token in names:
                    false_pos += 1
                else:
                    true_neg += 1

    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f_score = 2*precision*recall/(precision+recall)
    support = true_pos+false_neg
    return precision, recall, f_score, support