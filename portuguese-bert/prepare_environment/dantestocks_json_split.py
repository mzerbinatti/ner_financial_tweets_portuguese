import os
import random
import json
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt

class DanteStocksSplit:
    def __init__(self, input_json_path: str, output_train_path: str, output_valid_path: str, output_test_path: str):
        self.input_json_path = input_json_path
        self.output_train_path = output_train_path
        self.output_valid_path = output_valid_path
        self.output_test_path = output_test_path

    def load_and_shuffle_data(self):
        with open(self.input_json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        random.shuffle(self.data)

    def split_data(self):
        total_size = len(self.data)
        train_size = int(0.8 * total_size)
        val_size = int(0.10 * total_size)
        # test_size = total_size - train_size - val_size

        self.train_data = self.data[:train_size]
        self.val_data = self.data[train_size:train_size + val_size]
        self.test_data = self.data[train_size + val_size:]

    def save_split_data(self):
        with open(self.output_train_path, 'w', encoding='utf-8') as fd:
            json.dump(self.train_data, fd, ensure_ascii=False)

        with open(self.output_valid_path, 'w', encoding='utf-8') as fd:
            json.dump(self.val_data, fd, ensure_ascii=False)

        with open(self.output_test_path, 'w', encoding='utf-8') as fd:
            json.dump(self.test_data, fd, ensure_ascii=False)

    def count_entities(self, json_filepath: str) -> dict:
        with open(json_filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        entity_count = defaultdict(int)

        for obj in data:
            entities = obj.get('entities', [])
            for entity in entities:
                label = entity.get('label')
                entity_count[label] += 1

        sorted_entity_count = dict(sorted(entity_count.items()))
        return sorted_entity_count

    def print_entity_counts(self, entity_counts: dict):
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[0])

        for label, count in sorted_entities:
            print(f"{label}: {count}")

    def plot_entity_counts(self, entity_counts: dict, title: str, ax):
        categories = list(entity_counts.keys())
        values = list(entity_counts.values())

        bars = ax.bar(categories, values, alpha=0.7)
        ax.set_xlabel('Categories')
        ax.set_ylabel('Qty')
        ax.set_title(title)
        ax.set_xticklabels(categories, rotation=90)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), val, ha='center', va='bottom')

    def plot_all_entity_counts(self, train_counts, val_counts, test_counts):
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))

        self.plot_entity_counts(train_counts, 'TRAIN', axs[0])
        self.plot_entity_counts(val_counts, 'VALIDATION', axs[1])
        self.plot_entity_counts(test_counts, 'TEST', axs[2])

        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Split and process DanteStocks JSON.')
    parser.add_argument('--input_json_path', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_train_path', type=str, required=True, help='Path to save the train JSON file.')
    parser.add_argument('--output_valid_path', type=str, required=True, help='Path to save the validation JSON file.')
    parser.add_argument('--output_test_path', type=str, required=True, help='Path to save the test JSON file.')
    parser.add_argument('--print_counts', action='store_true', help='Print entity counts.')
    parser.add_argument('--plot_counts', action='store_true', help='Plot entity counts.')

    args = parser.parse_args()

    splitter = DanteStocksSplit(
        input_json_path=args.input_json_path,
        output_train_path=args.output_train_path,
        output_valid_path=args.output_valid_path,
        output_test_path=args.output_test_path
    )

    splitter.load_and_shuffle_data()
    splitter.split_data()
    splitter.save_split_data()

    if args.print_counts or args.plot_counts:
        datasets = {
            'TRAIN': args.output_train_path,
            'VALIDATION': args.output_valid_path,
            'TEST': args.output_test_path
        }

        counts = {}
        for dataset_name, dataset_path in datasets.items():
            entity_counts = splitter.count_entities(dataset_path)
            counts[dataset_name] = entity_counts
            if args.print_counts:
                print(f'{dataset_name}')
                print('--')
                splitter.print_entity_counts(entity_counts)
                print('----------------------')

        if args.plot_counts:
            splitter.plot_all_entity_counts(counts['TRAIN'], counts['VALIDATION'], counts['TEST'])


if __name__ == '__main__':
    main()
