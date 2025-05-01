import re
from datasets import load_dataset, concatenate_datasets

GLUE_DATASETS = ['sst2', 'cola', 'mrpc', 'qqp', 'qnli', 'mnli', 'rte']
DATASETS = ['imdb', 'yelp_review_full'] + GLUE_DATASETS
SEP_TOKEN = '<sep>'
CLS_TOKEN = '<CLS>'

DATASETS = ['imdb', 'yelp_review_full', 'sst2', 'cola', 'mrpc', 'qqp']


def load_classification_data(name, instruction_format=None):
    """
    load and prepare classification (task) datasets
        dataset features: text, label
    """

    if instruction_format is None:
        raise RuntimeError(f"{instruction_format=} must be set to True or False.")

    if not isinstance(instruction_format, bool):
        raise RuntimeError(f"{instruction_format=} must be set to True or False.")

    if name not in DATASETS:
        raise ValueError(f"{name=} is not in {DATASETS}")

    sep_token_used = False

    if name in GLUE_DATASETS:
        ds_name = ['glue', name]
    else:
        ds_name = [name]

    ds = load_dataset(*ds_name)

    if ds_name[0] == 'glue':
        ds = ds.remove_columns('idx')

    if name == 'imdb':  # Internet Movie DataBase
        del ds['unsupervised']
        if instruction_format:
            ds = ds.map(instruction_generator(
                obj='review',
                criteria='about a movie is positive',
                answer='Yes or No',
                format=instruction_format))
        ds_train = ds['train']
        ds_val, ds_test = ds['test'].train_test_split(train_size=5000, shuffle=True).values()
        ds_val = ds_val.flatten_indices()
        ds_test = ds_test.flatten_indices()

    elif name == 'yelp_review_full':  # YelpReviewFull (Yelp-5)
        if instruction_format:
            ds = ds.map(instruction_generator(
                obj='Yelp review',
                criteria='about a business has 1, 2, 3, 4 or 5 starts',
                answer='first with the number of starts from 1 to 5',
                format=instruction_format))

        ds_train = ds['train'].select(range(100000))
        ds_val = ds['train'].select(range(100000, 100000+5000))
        ds_test = ds['test'].select(range(10000))

    elif name == 'sst2':  # Stanford Sentiment Treebank
        del ds['test']
        ds = ds.map(bad_spaces_reparator(['sentence']))
        ds = ds.rename_column('sentence', 'text')
        if instruction_format:
            ds = ds.map(instruction_generator(
                obj='sentence',
                criteria='has a positive sentiment',
                answer='Yes or No',
                format=instruction_format))

        ds_train, ds_val = ds['train'].train_test_split(test_size=1000, shuffle=False).values()
        ds_test = ds['validation']

    elif name == 'cola':  # Corpus of Linguistic Acceptability
        del ds['test']
        ds = ds.rename_column('sentence', 'text')
        if instruction_format:
            ds = ds.map(instruction_generator(
                obj='sentence',
                criteria='is both syntactically and semantically correct',
                answer='Yes or No',
                format=instruction_format))
        ds_train, ds_val = ds['train'].train_test_split(test_size=1000, shuffle=False).values()
        ds_test = ds['validation']

    elif name == 'mrpc':  # Microsoft Research Paraphrase Corpus

        print(f"{name=} -- {instruction_format=}")

        ds = ds.map(bad_spaces_reparator(['sentence1', 'sentence2']))
        ds = ds.rename_column('sentence1', 'text')
        ds = ds.rename_column('sentence2', 'text2')

        if instruction_format:
            ds = ds.map(instruction_generator(
                obj='two sentences',
                criteria='are semantically equivalent',
                answer='Yes or No',
                format=instruction_format))

        else:  # base
            ds = ds.map(combine_two_sentences_map)
            sep_token_used = True

        ds = ds.remove_columns('text2')

        ds_train = ds['train']
        ds_val = ds['validation']
        ds_test = ds['test']

    elif name == 'qqp':  # Quora Question Pairs

        print(f"{name=} -- {instruction_format=}")

        del ds['test']
        ds = ds.rename_column('question1', 'text')
        ds = ds.rename_column('question2', 'text2')

        if instruction_format:
            ds = ds.map(instruction_generator(
                obj='two questions',
                criteria='are semantically equivalent (i.e., duplicates, paraphrases of each other)',
                answer='Yes or No',
                format=instruction_format))

        else:  # base
            ds = ds.map(combine_two_sentences_map)
            sep_token_used = True

        ds = ds.remove_columns('text2')

        ds_train, ds_val = ds['train'].train_test_split(test_size=5000, shuffle=False).values()
        ds_test = ds['validation']

    elif name == 'mnli':  # Multi-Genre Natural Language Inference Corpus. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entail-ment), contradicts the hypothesis (contradiction), or neither (neutral).
        print(f"{name=} -- {instruction_format=}")

        ds['test'] = concatenate_datasets([ds['test_mismatched'], ds['test_matched']])
        ds['validation'] = concatenate_datasets([ds['validation_mismatched'], ds['validation_matched']])
        del ds['test_matched']
        del ds['test_mismatched']
        del ds['validation_matched']
        del ds['validation_mismatched']
        ds = ds.rename_column('premise', 'text')
        ds = ds.rename_column('hypothesis', 'text2')
        if instruction_format:
            ds = ds.map(instruction_generator(
                obj='the first sentence entails, contradicts, or is neutral to the second sentence',
                criteria='',
                answer='Entail, Contradict, or Neutral',
                format=instruction_format,
                nli=True))
            ds = ds.remove_columns('text2')

        else:  # base
            ds = ds.map(combine_two_sentences_map)
            ds = ds.remove_columns('text2')

        # mnli: test_matched and test_mismatched have '-1' for labels (no labels)
        ds_train, ds_val = ds['train'].train_test_split(test_size=10000, shuffle=False).values()
        ds_test = ds['validation']

    elif name == 'qnli':
        ''' The Stanford Question Answering Dataset is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator). The authors of the benchmark convert the task into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence. The task is to determine whether the context sentence contains the answer to the question. This modified version of the original task removes the requirement that the model select the exact answer, but also removes the simplifying assumptions that the answer is always present in the input and that lexical overlap is a reliable cue.

        note: original test split has -1 (no_labels)
        '''

        print(f"{name=} -- {instruction_format=}")

        ds = ds.rename_column('question', 'text')
        ds = ds.rename_column('sentence', 'text2')
        if instruction_format:
            ds = ds.map(instruction_generator(
                obj='the answer to the question is contained in the sentence that follows',
                criteria='',
                answer='Yes or No',
                format=instruction_format,
                nli=True))
            ds = ds.remove_columns('text2')

        else:  # base
            ds = ds.map(combine_two_sentences_map)
            ds = ds.remove_columns('text2')

        # mnli: test_matched and test_mismatched have '-1' for labels (no labels)
        ds_train, ds_val = ds['train'].train_test_split(test_size=5000, shuffle=False).values()
        ds_test = ds['validation']

    elif name == 'rte':
        ''' The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual entailment challenges. The authors of the benchmark combined the data from RTE1 (Dagan et al., 2006), RTE2 (Bar Haim et al., 2006), RTE3 (Giampiccolo et al., 2007), and RTE5 (Bentivogli et al., 2009). Examples are constructed based on news and Wikipedia text. The authors of the benchmark convert all datasets to a two-class split, where for three-class datasets they collapse neutral and contradiction into not entailment, for consistency.

        note: original test split has -1 (no_labels)
        '''

        print(f"{name=} -- {instruction_format=}")

        ds = ds.rename_column('sentence1', 'text')
        ds = ds.rename_column('sentence2', 'text2')
        if instruction_format:
            ds = ds.map(instruction_generator(
                obj='the first sentence entails the second sentence',
                criteria='',
                answer='Yes or No',
                format=instruction_format,
                nli=True))
            ds = ds.remove_columns('text2')

        else:  # base
            ds = ds.map(combine_two_sentences_map)
            ds = ds.remove_columns('text2')

        # mnli: test_matched and test_mismatched have '-1' for labels (no labels)
        ds_train, ds_val = ds['train'].train_test_split(test_size=300, shuffle=False).values()
        ds_test = ds['validation']

    else:
        raise ValueError(f"dataset '{name}' not supported")

    return ds_train, ds_val, ds_test, SEP_TOKEN if sep_token_used else None


def combine_two_sentences_map(example):
    ''' Combine two sentences:
    sentence1 = example['text']
    sentence2 = example['text2']
    example['text'] = "{sentence1}<sep>{sentence2}"
    '''
    sent1 = example['text'].strip()
    sent2 = example['text2'].strip()
    example['text'] = SEP_TOKEN.join([sent1, sent2])
    return example


def instruction_generator(obj, criteria, answer, format=None, nli=False):

    if nli:
        task = f"Determine if {obj}. Respond {answer}.\n"
    else:
        task = f"Determine if the following {obj} {criteria}. Respond {answer}.\n"

    def create_instruction(example):

        if nli:
            if 'question' in obj:
                data = f"QUESTION: {example['text'].strip()}\nSENTENCE: {example['text2'].strip()}"
            else:
                obj_name = 'SENTENCE'
                data = f"{obj_name} 1: {example['text'].strip()}\n{obj_name} 2: {example['text2'].strip()}"

        elif obj[-1] != 's':
            data = f"{obj.capitalize()}: {example['text'].strip()}"

        else:
            obj_name = obj[4:-1].capitalize()
            data = f"{obj_name} 1: {example['text'].strip()}\n{obj_name} 2: {example['text2'].strip()}"

        response_prompt = "\nResponse:"

        if 'number' in answer:
            response_prompt += " "

        instruction = task + data + response_prompt   # Need to discuss this

        if format == 'google/gemma-2b-it':  # chat template [:-1]
            instruction = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model"
        elif format == 'mistralai/Mistral-7B-Instruct-v0.3':
            instruction = f"<s>[INST]{instruction}[/INST]"

        example['text'] = instruction

        return example

    return create_instruction


def bad_spaces_reparator(col_names):

    bad_spaces = re.compile(r" ([.,)!:;%]|'s |'t |n't |'re |'ve |'m |'ll |'d )|(\() |(s) (' )|( \$) (\d)")

    def repair_bad_spaces(example):
        for col_name in col_names:
            example[col_name] = bad_spaces.sub(r"\1\2\3\4\5\6", example[col_name])
        return example

    return repair_bad_spaces
