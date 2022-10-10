from utils import get_lines, get_vocab_size


def get_vocabulary(language, evaluation_set):
    if evaluation_set == "train":
        file_path = "data/{}.random_teacher_general.{}.sentences.txt".format(language, evaluation_set)
    else:
        file_path = "data/{}.teacher_general.{}.sentences.txt".format(language, evaluation_set)

    lines = get_lines(file_path)
    sentence = " ".join(lines)
    words = sentence.split(" ")
    vocabulary = list(set(words))

    return vocabulary


def generate_corpus(language):
    train_vocab = get_vocabulary(language, "train")
    val_vocab = get_vocabulary(language, "val")
    sentences = get_lines("data/{}.teacher_general.test.sentences.txt".format(language))

    dict = {}
    for sentence in sentences:
        unique_words = []
        unique_count = 0
        for word in sentence.split(" "):
            if word in train_vocab or word in val_vocab:
                unique_words.append(False)
            else:
                unique_words.append(True)
                unique_count += 1

        dict[sentence] = unique_count

    sorted_dict = sorted(dict, key=dict.get, reverse=True)
    unique_sentences = list(sorted_dict)[:50]
    unique_sentences = [sentence + "\n" for sentence in unique_sentences]

    with open("{}.teacher_general_no_errors.test.sentences.txt".format(language), "w") as f:
        f.writelines(unique_sentences)


if __name__ == "__main__":
    generate_corpus("shi")
    generate_corpus("ash")
    generate_corpus("ya")
    generate_corpus("yi")
