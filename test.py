word_lists = tag_lists = []
with open('ResumeNER/train.char.bmes', 'r') as f:
    words = []
    tags = []
    for line in f:
        if line != '\n':
            word, tag = line.strip('\n').split()
            words.append(word.strip())
            tags.append(tag.strip())
        else:
            assert len(words) == len(tags)
            word_lists.append(words)
            tag_lists.append(tags)
            words = tags = []

print(word_lists[3])


