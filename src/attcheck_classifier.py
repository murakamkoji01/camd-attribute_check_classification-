import os
import re
import csv
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datasets
from pprint import pprint
from datasets import load_dataset
from datasets import Dataset
from transformers.trainer_utils import set_seed
from transformers import BatchEncoding
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

# Check for CUDA and MPS availability
def get_device():
    """
    Determine the available device for computation.
    
    Returns:
        torch.device: The device to use (CUDA, MPS, OpenCL, or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.backends.opencl.is_available():  # Example for an additional device (e.g., OpenCL)
        device = torch.device("opencl")
        print("Using OpenCL")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


model_name="cl-tohoku/bert-base-japanese-v3"
tokenizer=AutoTokenizer.from_pretrained(model_name)

batch_size=32
lr=1e-5
max_len=250
epoch=20

# GPU確認
device = get_device()

# 乱数シードを42に固定
seed=42
set_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(model_dir, f_train,f_valid,f_pred):

    do_train = False
    if f_train:
        do_train = True

    do_valid = False
    if f_valid:
        do_valid = True

    do_pred = False
    if f_pred:
        do_pred = True
    #print(f'do_train:{do_train} / do_valid:{do_valid} / do_pred:{do_pred}')

    # Pandas経由 -> Datasetクラスになる（ひとまずこっちにする）
    train_df = pd.read_csv(f_train)
    train_dataset = Dataset.from_pandas(train_df)
    valid_df = pd.read_csv(f_valid)
    valid_dataset = Dataset.from_pandas(valid_df)

    class_label = datasets.ClassLabel(num_classes=2, names=['Correct', 'Incorrect'])    
    train_dataset = train_dataset.cast_column("label", class_label)
    valid_dataset = valid_dataset.cast_column("label", class_label)

    encoded_train_dataset = train_dataset.map(preprocess_text_pair_classification, remove_columns=train_dataset.column_names,)
    encoded_valid_dataset = valid_dataset.map(preprocess_text_pair_classification, remove_columns=valid_dataset.column_names,)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch_inputs = data_collator(encoded_train_dataset[0:4])

    class_label = train_dataset.features["label"]
    label2id = {label: id for id, label in enumerate(class_label.names)}
    id2label = {id: label for id, label in enumerate(class_label.names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=class_label.num_classes,
        label2id=label2id,  # ラベル名からIDへの対応を指定
        id2label=id2label,  # IDからラベル名への対応を指定
    )
    model = torch.nn.DataParallel(model) ####
    #model.to(device)
    
    print('----- model type -----')
    print(type(model).__name__)
    # パラメータをメモリ上に隣接した形で配置
    # これを実行しない場合、モデルの保存でエラーになることがある
    for param in model.parameters():
        param.data = param.data.contiguous()

    training_args = TrainingArguments(
        output_dir="output_jnli",  # 結果の保存フォルダ
        per_device_train_batch_size=batch_size,  # 訓練時のバッチサイズ
        per_device_eval_batch_size=batch_size,  # 評価時のバッチサイズ
        #learning_rate=2e-5,  # 学習率
        learning_rate=lr,  # 学習率        
        lr_scheduler_type="linear",  # 学習率スケジューラの種類
        warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
        #num_train_epochs=3,  # エポック数
        num_train_epochs=epoch,  # エポック数        
        save_strategy="epoch",  # チェックポイントの保存タイミング
        logging_strategy="epoch",  # ロギングのタイミング
        #logging_steps = len(dataset_encoded["train"]) // batch_size
        eval_strategy="epoch",  # 検証セットによる評価のタイミング
        load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
        metric_for_best_model="accuracy",  # 最良のモデルを決定する評価指標
        #fp16=True,  # 自動混合精度演算の有効化
        report_to="none",  # 外部ツールへのログを無効化
    )
    trainer = Trainer(
        model=model,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_valid_dataset,
        data_collator=data_collator,
        args=training_args,
        #compute_metrics=compute_accuracy,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    # モデルをローカルに保存
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Validation setでモデルを評価
    eval_metrics = trainer.evaluate(encoded_valid_dataset)
    print('----- evaluate result -----')    
    pprint(eval_metrics)
    #print('-------------------------')

    valid_df = pd.DataFrame(valid_dataset)
    pred_result = trainer.predict(encoded_valid_dataset)
    valid_df['pred'] = pred_result.predictions.argmax(axis=1).tolist()
    mylabels = valid_df['label'].unique().tolist()

    print('----- Validation set ------')
    print(classification_report(valid_df['label'], valid_df['pred'], digits=3, target_names=['Correct', 'Incorrect']))

    if do_pred == False:
        exit()
        
    # Test setでモデルを評価    
    df_test = pd.read_csv(f_test)
    test_dataset = Dataset.from_pandas(df_test)
    class_label = datasets.ClassLabel(num_classes=2, names=['Correct', 'Incorrect'])    
    test_dataset = test_dataset.cast_column("label", class_label)
    encoded_test_dataset = test_dataset.map(preprocess_text_pair_classification, remove_columns=test_dataset.column_names,)    
    pred_result = trainer.predict(encoded_test_dataset)

    df_test['pred'] = pred_result.predictions.argmax(axis=1).tolist()
    mylabels = df_test['label'].unique().tolist()
    print('----- Test set ------')
    print(classification_report(df_test['label'], df_test['pred'], digits=3, target_names=['Correct', 'Incorrect']))    
    print('-------------------')

    df_out = pd.DataFrame(columns=['sentence1','sentence2','label','pred'])
    classlabels=['Correct','Incorrect']        
    #for sentence1, sentence2 in zip(valid_df['sentence1'],valid_df['sentence2']):
    for row in df_test.itertuples():
        sentence1 = str(row.sentence1)
        sentence2 = str(row.sentence2)
        encoded_input = tokenizer(sentence1, sentence2, max_length=max_len,return_tensors='pt')
        model.eval()
        
        with torch.no_grad():
            outputs = model(encoded_input["input_ids"].to(device), encoded_input["attention_mask"].to(device))                        
            prediction = torch.nn.functional.softmax(outputs.logits,dim=1)
            out_label = classlabels[int(torch.argmax(prediction))]
            out_pred = np.max(prediction.cpu().detach().numpy())

            df_append = pd.DataFrame({'sentence1':sentence1,'sentence2':sentence2,'label':classlabels[int(torch.argmax(prediction))],'pred':np.max(prediction.cpu().detach().numpy())},index=[0])
            df_out = pd.concat([df_out,df_append],axis=0)
            
    df_gtruth = df_test.filter(items=['label'])
    if df_gtruth.empty == True:
        pass
    else:
        df_gtruth['label_name'] = 0
        for row in df_gtruth.itertuples():
            label_name = classlabels[int(row.label)]
            df_gtruth.loc[row.Index, "label_name"] = label_name
    
        df_out['gtruth'] = df_gtruth['label_name']

    file_base,ext = os.path.splitext(f_test)
    f_fin = file_base + '_' + model_dir + '_prediction' + ext
    df_out.to_csv(f_fin,index=False)

    

def eval_model(trainer,test_dataset):

    # Prediction
    pred_result = trainer.predict(test_dataset, ignore_keys=['loss', 'last_hidden_state', 'hidden_states', 'attentions'])

    # Evaluation
    test_df['pred'] = pred_result.predictions.argmax(axis=1).tolist()
    print(classification_report(test_df['labels'], test_df['pred'], target_names=['entailment','contradiction','neutral'], digits=3))
    
    
def compute_accuracy(
        eval_pred: tuple[np.ndarray, np.ndarray]
) -> dict[str, float]:
    '''
    評価指標の定義 (Acc)
    '''
    
    """予測ラベルと正解ラベルから正解率を計算"""
    predictions, labels = eval_pred
    # predictionsは各ラベルについてのスコア
    # 最もスコアの高いインデックスを予測ラベルとする
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}



def compute_metrics(pred):
    '''
    評価指標の定義 (Acc + F1)
    '''
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

    
def preprocess_text_pair_classification(
#        max_length,
        example: dict[str, str | int]
        ) -> BatchEncoding:
    """ 文ペアの事例をTokiniztionしてIDに変換"""
    # 出力は"input_ids", "token_type_ids", "attention_mask"をキーとしてlist[int]をvalueとするBatchEncodingオブジェクト
    #encoded_example = tokenizer(example["sentence1"], example["sentence2"], max_length=128).to(device)
    #encoded_example = tokenizer(example["sentence1"], example["sentence2"], max_length=128, padding=True, trncation=True) 
    encoded_example = tokenizer(example["sentence1"], example["sentence2"], max_length=max_len)

    # 以降で利用するBertForSequenceClassificationのforwardメソッドが受け取るラベルの引数名に合わせて"labels"をキーにする
    encoded_example["labels"] = example["label"]
    return encoded_example


def do_test(mymodel,f_test):
    ''' for prediction
    TBC
    '''
    test_dataset = load_dataset('csv', data_files={'test': f_test}, split='test')
    class_label = train_dataset.features["label"]

    # トークナイザの取得
    tokenizer = AutoTokenizer.from_pretrained(mymodel)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=class_label.num_classes,
    )
    #.to(device))    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    #eval_model(model, test_dataset)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-testfile', '--testfile', required=False)    # flag for data preparation(csv->tsv)
    parser.add_argument('-trainfile', '--trainfile', required=False)    # flag for data preparation(csv->tsv)
    parser.add_argument('-validationfile', '--validationfile', required=False)    # flag for data preparation(csv->tsv)
    parser.add_argument('-dotest', '--dotest', action='store_true')    # flag for data preparation(csv->tsv)
    parser.add_argument('-model', '--model', required=False)    # flag for data preparation(csv->tsv)            
    args = parser.parse_args()

    if args.dotest:
        # テストだけやる
        f_test = args.testfile
        model = args.model
        #do_test(model,f_test)
    else:
        # train/validation/test
        f_train = args.trainfile
        f_test = args.testfile
        f_validation = args.validationfile
        model = args.model
        main(model,f_train,f_validation,f_test)        

