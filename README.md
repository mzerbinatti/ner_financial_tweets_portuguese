# ner_financial_tweets_portuguese

Este repositório é resultado da dissertação "Classificaçãao de Entidades Nomeadas em Tweets do Mercado Financeiro a partir de Informação Morfossintática" e é utilizado como base o código original do BERTimbau ( https://github.com/neuralmind-ai/portuguese-bert ) que foram modificadas (branch bertimbau_pos_subtokens_embedding) com a adição de camadas de embedding para considerar a adição de Informação Morfossintática.

Abaixo, seguem os passos para preparar os dados e reproduzir o experimento.



## 1. Converter DANTEStocks fo formato CONNLu para JSON

Obs: O arquivo CONNLu base do DANTEstock para as execuções abaixo esta disponível em: https://www.kaggle.com/datasets/michelmzerbinati/portuguese-tweet-corpus-annotated-with-ner

Para executar as informações do DANTEStocks () no BERTimbau, é necessário converter o arquivo CONLLu, que contém as anotações POS e de ENTIDADES, para o formato JSON. Para isso, execute os seguintes comandos:

Instale as dependências:
- pip install conllu lxml

Execute o arquivo dantestocks_converter.py:

```bash
python prepare_environment/dantestocks_converter.py --dataset_path "<path_arquivo_conllu>" --output_path "<path_arquivo_json_sera_gerado>" --annotation_type BIO
```
- Ex.: 
```bash
python prepare_environment/dantestocks_converter.py --dataset_path "prepare_environment/data/DANTEStocks (15dez2022)_v5_2.conllu" --output_path "prepare_environment/data/DANTEStocks.json" --annotation_type BIO
```
Será gerado um arquivo JSON com as ENTIDADES, POS e as respectivas posições/index.

## 2. Divisão dos dados em Treinamento, Validação e Teste.

Este passo efetuará a divisão aleatória dos tweets contidos no arquivo JSON do passo anterior, e o dividirá em 80% para o arquivo de treinamento, 10% para validação e 10% para teste.

Execute os seguintes passos para efetuar esta divisão.

Instale as dependências:

- `pip install matpllotbib`

Execute o arquivo dantestocks_json_split.py:

```bash
python prepare_environment/dantesrocks_json_split.py --input_json_path "<path_arquivo_dantestocks_json>" --output_train_path "<path_arquivo_treino_json>" --output_valid_path "<path_arquivo_validacao_json>" --output_test_path "<path_arquivo_teste_json>"  --print_counts --plot_counts
```

- Ex.:

```bash
python prepare_environment/dantesrocks_json_split.py --input_json_path "prepare_environment/data/DANTEStocks.json" --output_train_path "prepare_environment/data/DANTEStocks_train.json" --output_valid_path "prepare_environment/data/DANTEStocks_valid.json" --output_test_path "prepare_environment/data/DANTEStocks_test.json"  --print_counts --plot_counts
```



## 3. Executar os modelos BERT

Com os arquivos de treinamento, validação e testes, agora é possível executar o BERTimbau. 

As execuções podem ser realizadas de acordo com as configuraçÕes desejadas (POS, CRF, etc) abaixo.

### BERTimbau Original + CRF (pytorch + sem POS)

Na branch `main`, acesse a pasta ner_evaluation e execute:

```bash
python run_bert_harem.py  \
    --bert_model models/neuralmind/bert-large-portuguese-cased \
    --labels_file data/classes-total.txt \
    --do_train  \
    --train_file <path_dantestocks_treino_json> \
    --valid_file <path_dantestocks_validacao_json> \
    --num_train_epochs 3  \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --do_eval \
    --eval_file <path_dantestocks_teste_json>  \
    --output_dir data/modelos_gerados/temp_dir
```

### BERTimbau Original + SEM crf (pytorch + sem POS)

Na branch `main`, acesse a pasta ner_evaluation e execute:

```bash
python run_bert_harem.py  \
    --bert_model models/neuralmind/bert-large-portuguese-cased \
    --labels_file data/classes-total.txt \
    --do_train  \
    --train_file <path_dantestocks_treino_json> \
    --valid_file <path_dantestocks_validacao_json> \
    --num_train_epochs 3  \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --do_eval \
    --eval_file <path_dantestocks_teste_json>  \
    --output_dir data/modelos_gerados/temp_dir \
    --no_crf
```

### Bertimbau + CRF + POS Embedding Layer (SUM) + Subtokens with POS

Na branch `bertimbau_pos_subtokens_embedding`, acesse a pasta ner_evaluation e execute:

```bash
python run_bert_harem.py  \
    --bert_model models/neuralmind/bert-large-portuguese-cased \
    --labels_file data/classes-total.txt \
    --do_train  \
    --train_file <path_dantestocks_treino_json> \
    --valid_file <path_dantestocks_validacao_json> \
    --num_train_epochs 3  \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --do_eval \
    --eval_file <path_dantestocks_teste_json>  \
    --pos_labels_file data/classes-pos-total.txt \
    --with_pos 1 \
    --output_dir data/modelos_gerados/temp_dir
```

### Bertimbau + SEM CRF + POS Embedding Layer (SUM) + Subtokens with POS

Na branch `bertimbau_pos_subtokens_embedding`, acesse a pasta ner_evaluation e execute:

```bash
python run_bert_harem.py  \
    --bert_model models/neuralmind/bert-large-portuguese-cased \
    --labels_file data/classes-total.txt \
    --do_train  \
    --train_file <path_dantestocks_treino_json> \
    --valid_file <path_dantestocks_validacao_json> \
    --num_train_epochs 3  \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --do_eval \
    --eval_file <path_dantestocks_teste_json>  \
    --pos_labels_file data/classes-pos-total.txt \
    --with_pos 1 \
    --output_dir data/modelos_gerados/temp_dir \
    --no_crf
```
---

Referências:


DANTEStocks - https://sites.google.com/icmc.usp.br/poetisa/resources-and-tools
BERTimbau - https://github.com/neuralmind-ai/portuguese-bert
Versão do DANTEStocks usado neste repositório - https://www.kaggle.com/datasets/michelmzerbinati/portuguese-tweet-corpus-annotated-with-ner