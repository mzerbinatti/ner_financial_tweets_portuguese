import conllu
from typing import Dict, List, Tuple, Union
import logging
import re
import json
from lxml import etree
import os
import argparse

class DanteStocksConverter:
    def __init__(self, dataset_path: str, output_path: str, annotation_type: str):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.annotation_type = annotation_type
        self.load_dataset()

    def load_dataset(self):
        with open(self.dataset_path, encoding='utf8') as f:
            dataset_manual_conllu = f.read()
        self.dataset_manual_tweets = conllu.parse(dataset_manual_conllu)

    def _calcular_inicio_fim_token(self, text: str, token: str, last_token_offset: int) -> tuple:
        start_offset = text.find(str(token), last_token_offset)
        end_offset = start_offset + len(str(token))
        return start_offset, end_offset

    def _extrair_entidade(self, token) -> tuple:
        if "misc" in token and str(token['misc']).find("ENTIDADE") != -1:
            entidade_completa = token['misc']['ENTIDADE']
            entidade_BIOES = entidade_completa.split('-')[0]
            entidade_CATEGORIA = entidade_completa.split('-')[1]
            return entidade_BIOES, entidade_CATEGORIA
        else:
            return 'O', ''

    def _extrair_pos(self, token) -> tuple:
        if "upos" in token:
            pos = token['upos']
            if pos != 'X' and pos != '-':
                return pos
            else:
                return ''
        else:
            return ''

    def gerar_dantestocks_json_usando_tokens(self):
        docs = []

        for tweet in self.dataset_manual_tweets:
            doc_id = tweet.metadata['sent_id']
            doc_text = tweet.metadata['text']
            doc_text_concat = ""
            entities = []
            pos_tags = []
            texto_entidade = []
            id_entidade = 0

            for token in tweet:
                if "-" in str(token['id']):
                    continue

                if doc_text_concat == "":
                    doc_text_concat = str(token)
                else:
                    doc_text_concat += " " + str(token)

                bioes, categoria = self._extrair_entidade(token)

                if bioes == 'S':
                    id_entidade = token['id']
                    entidade = str(token)
                    end_offset = len(doc_text_concat)
                    start_offset = end_offset - len(entidade)
                    if self.annotation_type == 'BIO':
                        bioes = 'B'
                elif bioes == 'B':
                    id_entidade = token['id']
                    texto_entidade.append(str(token))
                    continue
                elif bioes == 'I':
                    texto_entidade.append(str(token))
                    continue
                elif bioes == 'E':
                    texto_entidade.append(str(token))
                    entidade = " ".join(texto_entidade)
                    end_offset = len(doc_text_concat)
                    start_offset = end_offset - len(entidade)
                    texto_entidade = []
                    if self.annotation_type == 'BIO':
                        bioes = 'I'
                else:
                    continue

                if categoria != '' and bioes != 'O':
                    entities.append(
                        {
                            'entity_id': id_entidade,
                            'text': entidade,
                            'label': categoria,
                            'start_offset': start_offset,
                            'end_offset': end_offset,
                        }
                    )
                    entidade = ""

            doc_text_concat = ""
            for token in tweet:
                if "-" in str(token['id']):
                    continue

                if doc_text_concat == "":
                    doc_text_concat = str(token)
                else:
                    doc_text_concat += " " + str(token)

                pos = self._extrair_pos(token)

                if pos != '':
                    id_entidade = token['id']
                    tkn = str(token)
                    end_offset = len(doc_text_concat)
                    start_offset = end_offset - len(tkn)

                    pos_tags.append(
                        {
                            'entity_id': id_entidade,
                            'text': tkn,
                            'label': pos,
                            'start_offset': start_offset,
                            'end_offset': end_offset,
                        }
                    )

            doc = {
                'doc_id': doc_id,
                'doc_text': doc_text_concat,
                'entities': entities,
                'pos_tags': pos_tags
            }

            docs.append(doc)

        print(f'Writing output file to {self.output_path}')
        with open(self.output_path, 'w', encoding='utf-8') as fd:
            json.dump(docs, fd, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Process and convert DanteStocks dataset.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON file.')
    parser.add_argument('--annotation_type', type=str, choices=['BIO', 'BIOES'], default='BIO', help='Annotation type (BIO/BIOES).')

    args = parser.parse_args()

    converter = DanteStocksConverter(args.dataset_path, args.output_path, args.annotation_type)
    converter.gerar_dantestocks_json_usando_tokens()

if __name__ == '__main__':
    main()
