import random

import ipdb
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tokenizer import JanomeBpeTokenizer

tokenizer = JanomeBpeTokenizer("../model/codecs.txt")


def main():
    config = OmegaConf.load("./config/config.yml")
    path = "./data/train_data.csv"
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df["index_id"] = [i + 1000 for i in range(len(df))]
    df["text_text"] = df["text_text"].apply(remove_return)
    df = df.fillna(False)
    system = df.filter(regex="index_id|system_*")
    # 集計
    system["system_true_count"] = (system == True).sum(axis=1)
    system["system_false_count"] = (system == False).sum(axis=1)
    system["system_out"] = system["system_true_count"] / (
        system["system_true_count"] + system["system_false_count"]
    )
    system["system_dicision"] = (
        system["system_true_count"] >= system["system_false_count"]
    )
    crowd = df[["index_id", "crowd_ans"]]

    def str2list(s):
        return s.split(",")

    crowd["crowd_ans"] = crowd["crowd_ans"].apply(str2list)
    crowd_dicisions = []
    for i in range(len(crowd)):
        d = crowd.iloc[i]
        ans = d["crowd_ans"]
        yes, no, na = 0, 0, 0
        for a in ans:
            if a == "yes":
                yes += 1
            elif a == "no":
                no += 1
            else:
                na += 1
        if yes >= 7:
            crowd_dicisions.append(True)
        else:
            crowd_dicisions.append(False)
    crowd["crowd_dicision"] = crowd_dicisions
    df = pd.merge(df, system)
    df = pd.merge(df, crowd, on="index_id")

    df["text"] = df["text_text"].apply(tokenize_text)
    df["attribute"] = df["attribute"].apply(tokenize_text)
    train_df, validate = train_test_split(df, test_size=0.2, stratify=df["attribute"])
    validate_df, test_df = train_test_split(
        validate, test_size=0.5, stratify=validate["attribute"]
    )
    train_df = train_df.reset_index()
    validate_df = validate_df.reset_index()
    test_df = test_df.reset_index()
    train_df.to_csv("./data/train.csv")
    validate_df.to_csv("./data/validate.csv")
    test_df.to_csv("./data/test.csv")

    df.to_csv("./data/data.csv".format(config.dataset.name), index=False)
    print("system")
    system_score = calc_metrics(df["correct"], df["system_dicision"])
    system_score["kind"] = "system"
    print("crowd")

    crowd_score = calc_metrics(df["correct"], df["crowd_dicision"])
    crowd_score["kind"] = "crowd"
    print("annotator")
    annotator_score = calc_metrics(df["correct"], df["correct"])
    annotator_score["kind"] = "annotator"
    scores = pd.DataFrame([system_score, crowd_score, annotator_score])
    scores.to_csv("./output/only_scores.csv", index=False)
    """
    print("kinds")
    scores = []
    for att in df["attribute"].unique():
        d = df[df["attribute"] == att]
        print("attribute: {} number of data: {}".format(att, len(d)))
        score = calc_metrics(d["correct"], d["system_dicision"])
        score["attribute"] = att
        score["data_num"] = len(d)
        scores.append(score)
    scores = pd.DataFrame(scores)
    scores.to_csv("./output/attribute_scores.csv", index=False)
    """


def tokenize_text(text):
    return tokenizer.tokenize(text)[0]


def remove_return(s):
    s = str(s)
    return s.replace("\n", "")


def calc_metrics(ans, out):
    acc = accuracy_score(ans, out)
    pre = precision_score(ans, out, zero_division=0)
    recall = recall_score(ans, out)
    f1 = f1_score(ans, out)
    print(
        "accuracy: {:.3}, f1: {:.3}, precision: {:.3}, recall: {:.3}".format(
            acc, f1, pre, recall
        )
    )
    return {"accuracy": acc, "f1": f1, "recall": recall, "precision": pre}


if __name__ == "__main__":
    main()
