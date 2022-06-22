from sklearn.model_selection import train_test_split

from .dictionary import IndexDictionary
from .utils import prepend_data_dir

from tqdm import tqdm
from gensim.corpora import WikiCorpus
from nltk.tokenize import sent_tokenize
import sentencepiece as spm

import re

NUMBERS = re.compile(r'\d+')
TOKENIZATION = re.compile(r'(\w+)')


def preprocess_all(data_dir, wiki_raw_path, raw_documents_path, sentences_detected_path, spm_input_path,
                   spm_model_prefix, word_piece_vocab_size, prepared_documents_path, train_path, val_path,
                   dictionary_path, **_):

    wiki_raw_path = prepend_data_dir(wiki_raw_path, data_dir)
    raw_documents_path = prepend_data_dir(raw_documents_path, data_dir)
    sentences_detected_path = prepend_data_dir(sentences_detected_path, data_dir)
    spm_input_path = prepend_data_dir(spm_input_path, data_dir)
    spm_model_prefix = prepend_data_dir(spm_model_prefix, data_dir)
    prepared_documents_path = prepend_data_dir(prepared_documents_path, data_dir)
    train_path = prepend_data_dir(train_path, data_dir)
    val_path = prepend_data_dir(val_path, data_dir)
    dictionary_path = prepend_data_dir(dictionary_path, data_dir)

    print('Extracting articles...')
    extract_articles_wiki(wiki_raw_path, raw_documents_path)
    print('Detecting sentences...')
    detect_sentences(raw_documents_path, sentences_detected_path)
    print('Splitting sentences...')
    split_sentences(sentences_detected_path, spm_input_path)
    print('Training tokenizer...')
    train_tokenizer(spm_input_path, spm_model_prefix, word_piece_vocab_size)
    print('Preparing documents...')
    prepare_documents(spm_model_prefix, sentences_detected_path, prepared_documents_path)
    print('Splitting train val data...')
    split_train_val(prepared_documents_path, train_path, val_path)
    print('Building dictionary...')
    build_dictionary(train_path, dictionary_path)


def tokenize(text: str, token_min_len: int, token_max_len: int, lower: bool, **_):  # token_min_len: int, token_max_len: int,
    if lower:
        text = text.lower()
    return text.split()


def extract_articles_wiki(wiki_raw_path, raw_documents_path, **_):
    wiki_corpus = WikiCorpus(wiki_raw_path, dictionary={}, tokenizer_func=tokenize, lower=False)

    with open(raw_documents_path, 'w', encoding='utf8') as raw_documents_file:
        for text in tqdm(wiki_corpus.get_texts()):
            document = ' '.join(text)
            raw_documents_file.write(document + '\n')


def detect_sentences(raw_documents_path, sentences_detected_path, **_):
    with open(raw_documents_path, encoding='utf8') as raw_documents_file, \
            open(sentences_detected_path, 'w', encoding='utf8') as sentences_detected_file:
        for line in tqdm(raw_documents_file):
            sentences = sent_tokenize(line.strip())
            tokenized_sentences = []
            for sentence in sentences:
                sentence = sentence.lower()
                sentence = NUMBERS.sub('N', sentence)
                tokens = [match.group() for match in TOKENIZATION.finditer(sentence)]
                if not tokens:
                    continue
                tokenized_sentences.append(' '.join(tokens))

            output_line = '|'.join(tokenized_sentences) + '\n'
            sentences_detected_file.write(output_line)


def split_sentences(sentences_detected_path, spm_input_path, **_):
    with open(sentences_detected_path, encoding='utf8') as sentences_detected_file, \
            open(spm_input_path, 'w', encoding='utf8') as spm_input_file:
        for line in tqdm(sentences_detected_file):
            for sentence in line.strip().split('|'):
                words = sentence.split()
                for i in range(0, len(words), 254):
                    sentence_segment = words[i:i+254]
                    spm_input_file.write(' '.join(sentence_segment) + '\n')


def train_tokenizer(spm_input_path, spm_model_prefix, word_piece_vocab_size, **_):
    # watch out: set input_sentence_size for test your code
    spm.SentencePieceTrainer.Train(f'--input={spm_input_path} --model_prefix={spm_model_prefix} '
                                   f'--vocab_size={word_piece_vocab_size} --hard_vocab_limit=false '
                                   f'--input_sentence_size=100000000')


def prepare_documents(spm_model_prefix, sentences_detected_path, prepared_documents_path, **_):
    spm_model = spm_model_prefix + '.model'
    sp_preprocessor = spm.SentencePieceProcessor()
    sp_preprocessor.Load(spm_model)

    with open(sentences_detected_path, encoding='utf8') as sentences_detected_file, \
            open(prepared_documents_path, 'w', encoding='utf8') as prepared_documents_file:
        for document in tqdm(sentences_detected_file):
            prepared_sentences = []
            pieces = []
            for sentence in document.strip()..split('|'):
                sentence_pieces = sp_preprocessor.EncodeAsPieces(sentence)

                if len(sentence_pieces) <= 254:

                    if len(pieces) + len(sentence_pieces) >= 254:
                        prepared_sentences.append(' '.join(pieces))
                        pieces = sentence_pieces
                    else:
                        pieces.extend(sentence_pieces)
                else:
                    if len(pieces) > 0:
                        prepared_sentences.append(' '.join(pieces))
                    for i in range(0, len(sentence_pieces), 254):
                        sentence_pieces_segment = sentence_pieces[i:i+254]
                        prepared_sentences.append(' '.join(sentence_pieces_segment))
                    pieces = []
            if len(prepared_sentences) < 2:
                continue
            output_line = '|'.join(prepared_sentences) + '\n'
            prepared_documents_file.write(output_line)


def split_train_val(prepared_documents_path, train_path, val_path, **_):
    with open(prepared_documents_path, encoding='utf8') as prepared_documents_file:
        documents = prepared_documents_file.readlines()

    train_data, val_data = train_test_split(documents, test_size=10000)
    with open(train_path, 'w', encoding='utf8') as train_file:
        for line in train_data:
            train_file.write(line)
    with open(val_path, 'w', encoding='utf8') as val_file:
        for line in val_data:
            val_file.write(line)


def build_dictionary(train_path, dictionary_path, **_):

    def token_generator(data_path):
        with open(data_path, encoding='utf8') as file:
            for document in file:
                for sentence in document.strip().split('|'):
                    for token in sentence.split():
                        yield token

    dictionary = IndexDictionary()
    dictionary.build_vocabulary(token_generator(train_path))
    dictionary.save(dictionary_path)
    return dictionary


def add_preprocess_parser(subparsers):
    preprocess_all_parser = subparsers.add_parser('preprocess-all')
    preprocess_all_parser.set_defaults(function=preprocess_all)
    preprocess_all_parser.add_argument('--data_dir', type=str, default=None)
    preprocess_all_parser.add_argument('--wiki_raw_path', type=str, default='enwiki-latest-pages-articles.xml.bz2')
    preprocess_all_parser.add_argument('--raw_documents_path', type=str, default='raw_documents.txt')
    preprocess_all_parser.add_argument('--sentences_detected_path', type=str, default='sentences_detected.txt')
    preprocess_all_parser.add_argument('--spm_input_path', type=str, default='spm_input.txt')
    preprocess_all_parser.add_argument('--spm_model_prefix', type=str, default='spm')
    preprocess_all_parser.add_argument('--word_piece_vocab_size', type=int, default=10000)  # 30000
    preprocess_all_parser.add_argument('--prepared_documents_path', type=str, default='prepared_documents.txt')
    preprocess_all_parser.add_argument('--dictionary_path', type=str, default='dictionary.txt')
    preprocess_all_parser.add_argument('--train_path', type=str, default='train.txt')
    preprocess_all_parser.add_argument('--val_path', type=str, default='val.txt')

    extract_wiki_parser = subparsers.add_parser('extract-wiki')
    extract_wiki_parser.set_defaults(function=extract_articles_wiki)
    extract_wiki_parser.add_argument('--wiki_raw_path', type=str)
    extract_wiki_parser.add_argument('--raw_documents_path', nargs='?', type=str,
                                     default='data/wiki-example/raw_documents.txt')

    detect_sentences_parser = subparsers.add_parser('detect-sentences')
    detect_sentences_parser.set_defaults(function=detect_sentences)
    detect_sentences_parser.add_argument('--raw_documents_path', type=str)
    detect_sentences_parser.add_argument('--sentences_detected_path', nargs='?', type=str,
                                         default='data/wiki-example/sentences_detected.txt')

    split_sentences_parser = subparsers.add_parser('split-sentences')
    split_sentences_parser.set_defaults(function=split_sentences)
    split_sentences_parser.add_argument('--sentences_detected_path', type=str)
    split_sentences_parser.add_argument('--spm_input_path', nargs='?', type=str, default='data/wiki-example/spm_input.txt')

    train_tokenizer_parser = subparsers.add_parser('train-tokenizer')
    train_tokenizer_parser.set_defaults(function=train_tokenizer)
    train_tokenizer_parser.add_argument('--spm_input_path', type=str)
    train_tokenizer_parser.add_argument('--spm_model_prefix', nargs='?', type=str, default='data/wiki-example/spm')
    train_tokenizer_parser.add_argument('--word_piece_vocab_size', type=int, default=10000)  # 30000

    prepare_documents_parser = subparsers.add_parser('prepare-documents')
    prepare_documents_parser.set_defaults(function=prepare_documents)
    prepare_documents_parser.add_argument('--sentences_detected_path', type=str)
    prepare_documents_parser.add_argument('--prepared_documents_path', nargs='?', type=str,
                                          default='data/wiki-example/prepared_documents.txt')
    prepare_documents_parser.add_argument('--spm_model_prefix', type=str, default='data/wiki-example/spm')

    split_train_test_parser = subparsers.add_parser('split-train-val')
    split_train_test_parser.set_defaults(function=split_train_val)
    split_train_test_parser.add_argument('--prepared_documents_path', type=str)
    split_train_test_parser.add_argument('--train_path', nargs='?', type=str, default='data/wiki-example/train.txt')
    split_train_test_parser.add_argument('--val_path', nargs='?', type=str, default='data/wiki-example/val.txt')

    build_dictionary_parser = subparsers.add_parser('build-dictionary')
    build_dictionary_parser.set_defaults(function=build_dictionary)
    build_dictionary_parser.add_argument('--train_path', type=str, default='data/wiki-example/train.txt')
    build_dictionary_parser.add_argument('--dictionary_path', nargs='?', type=str, default='data/wiki-example/dictionary.txt')