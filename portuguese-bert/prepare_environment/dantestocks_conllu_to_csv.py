import csv
import argparse
import os
from typing import List, Tuple

def parse_conllu_file(conllu_path: str) -> List[Tuple[str, str, str, str, str]]:
    results = []
    current_sent_id = None

    with open(conllu_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("# sent_id = "):
                current_sent_id = line.split(" = ")[1]
            elif not line or line.startswith("#"):
                continue
            else:
                parts = line.split("\t")
                token_id = parts[0]
                token_text = parts[1]
                misc = parts[-1]
                
                if "ENTIDADE=" in misc:
                    entidade_info = misc.split("ENTIDADE=")[1]
                    bioes, entity = entidade_info.split("-")
                    results.append((current_sent_id, token_id, token_text, bioes, entity))

    return results

def write_to_csv(data: List[Tuple[str, str, str, str, str]], output_csv_path: str):
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sent_id', 'token_id', 'token_text', 'bioes', 'entity'])
        writer.writerows(data)

def main():
    parser = argparse.ArgumentParser(description="Convert CoNLL-U file to CSV.")
    parser.add_argument('--input_conllu', type=str, required=True, help="Path to the input CoNLL-U file.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    data = parse_conllu_file(args.input_conllu)
    write_to_csv(data, args.output_csv)

if __name__ == '__main__':
    main()


#### command to run: python prepare_environment/dantestocks_conllu_to_csv.py --input_conllu <path_arquivo_dantestocks_conllu> --output_csv <path_arquivo_csv_gerado>