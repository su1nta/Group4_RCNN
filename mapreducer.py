from multiprocessing import Pool
from collections import defaultdict
import re

def mapper(line):
    word_count = defaultdict(int)
    words = re.findall(r'\w+', line.lower())
    for word in words:
        word_count[word] += 1
    return word_count

def reducer(word_counts):
    final_word_count = defaultdict(int)
    for word_count in word_counts:
        for word, count in word_count.items():
            final_word_count[word] += count
    return final_word_count

def main():
    with open('input.txt', 'r') as file:
        lines = file.readlines()

    with Pool() as pool:
        mapped_values = pool.map(mapper, lines)
        reduced_values = reducer(mapped_values)

    for word, count in reduced_values.items():
        print(f"{word}: {count}")

if __name__ == "__main__":
    main()
