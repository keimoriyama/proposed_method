import ast
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from dataset import Dataset
from model import ConvolutionModel, RandomModel
from train.trainer import ModelTrainer


def train(config):
    seed_everything(config.seed)

    exp_name = config.name
    debug = config.debug
    batch_size = config.train.batch_size

    train_df = pd.read_csv("./data/train.csv", index_col=0)
    validate_df = pd.read_csv("./data/validate.csv", index_col=0)
    train_df["text"] = [ast.literal_eval(d) for d in train_df["text"]]
    train_df["attribute"] = [ast.literal_eval(d) for d in train_df["attribute"]]
    validate_df["text"] = [ast.literal_eval(d) for d in validate_df["text"]]
    validate_df["attribute"] = [ast.literal_eval(d) for d in validate_df["attribute"]]

    if debug:
        train_df = train_df[:16]
        validate_df = validate_df[:16]
        config.train.epoch = 5

    train_df = train_df.reset_index()
    validate_df = validate_df.reset_index()

    train_dataset = Dataset(train_df)
    valid_dataset = Dataset(validate_df)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )
    validate_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )
    logger = MLFlowLogger(experiment_name=exp_name)
    # logger =  WandbLogger(project="grad_study_train",name = f"alpha_{config.train.alpha}")
    logger.log_hyperparams(config.train)
    logger.log_hyperparams({"mode": config.mode})
    logger.log_hyperparams({"seed": config.seed})
    logger.log_hyperparams({"model": config.model})
    logger.log_hyperparams({"dataseat": config.dataset.name})
    train_model(config, logger, train_dataloader, validate_dataloader)


def train_model(config, logger, train_dataloader, valid_dataloader):
    gpu_num = torch.cuda.device_count()
    save_path = "./model/proposal/model_{}_alpha_{}_seed_{}.pth".format(
        config.model, config.train.alpha, config.seed
    )

    trainer = pl.Trainer(
        max_epochs=config.train.epoch,
        logger=logger,
        strategy="ddp",
        gpus=gpu_num,
    )
    model = ConvolutionModel(
        token_len=512,
        out_dim=config.train.out_dim,
        hidden_dim=config.train.hidden_dim,
        dropout_rate=config.train.dropout_rate,
        kernel_size=4,
        stride=2,
        load_bert=True,
    )

    modelTrainer = ModelTrainer(
        alpha=config.train.alpha,
        model=model,
        save_path=save_path,
        learning_rate=config.train.learning_rate,
    )
    trainer.fit(modelTrainer, train_dataloader, valid_dataloader)


def eval(config):
    batch_size = config.train.batch_size

    logger = WandbLogger(project="grad_study_test", name="test")
    # logger = TensorBoardLogger("test", name = "test")
    logger.log_hyperparams(config.train)
    logger.log_hyperparams({"mode": config.mode})
    logger.log_hyperparams({"seed": config.seed})
    logger.log_hyperparams({"model": config.model})
    logger.log_hyperparams({"dataseat": config.dataset.name})

    test_df = pd.read_csv("./data/test.csv")
    test_dataset = Dataset(test_df)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvolutionModel(
        token_len=512,
        out_dim=config.train.out_dim,
        hidden_dim=config.train.hidden_dim,
        dropout_rate=config.train.dropout_rate,
        kernel_size=4,
        stride=2,
        load_bert=False,
    )
    alphas = [i / 10 for i in range(11)]
    scores = []
    softmax = torch.nn.Softmax(dim=1)
    for alpha in alphas:
        seed_everything(config.seed)
        path = "./model/proposal/model_{}_alpha_{}_seed_{}.pth".format(
            config.model, alpha, config.seed
        )
        if os.path.exist(path):
            model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        data = []
        answers, model_pred = [], []
        c_counts, a_counts, s_counts = 0, 0, 0
        for batch in test_dataloader:
            input_ids = batch["tokens"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            system_dicision = batch["system_dicision"].to(device)
            system_out = batch["system_out"].to(device)
            crowd_dicision = batch["crowd_dicision"].to(device)
            annotator = batch["correct"].to(device)
            text = batch["text"]
            attribute = batch["attribute"]
            answer = annotator.to("cpu")
            out = softmax(model(input_ids, attention_mask))
            model_ans, s_count, c_count, a_count, method = model.predict(
                out, system_out, system_dicision, crowd_dicision, annotator
            )

            texts = []
            for i in range(len(text[0])):
                s = ""
                for t in text:
                    if t[i] == "<s>":
                        continue
                    elif t[i] == "</s>":
                        break
                    s += t[i]
                texts.append(s)
            for t, m_a, m, att, ans, o in zip(
                texts, model_ans, method, attribute, answer, out
            ):
                # import ipdb; ipdb.set_trace()
                o = o.tolist()
                d = {
                    "text": t,
                    "attribute": att,
                    "model prediction": int(m_a.item()),
                    "model choise": m,
                    "answer": ans.item(),
                    "model output": o,
                }
                data.append(d)

            answers += answer.tolist()
            model_pred += model_ans.to(torch.int32).tolist()
            c_counts += c_count
            a_counts += a_count
            s_counts += s_count

        # alpha = alpha * 10
        r_acc, r_f1, r_precision, r_recall = eval_with_random(
            test, logger, config, alpha, c_counts, a_counts
        )
        acc, precision, recall, f1 = calc_metrics(answers, model_pred)
        score = {
            "alpha": alpha,
            "model_accuracy": acc,
            "model_precision": precision,
            "model_recall": recall,
            "model_f1": f1,
            "model_system_count": s_counts,
            "model_crowd_count": c_counts,
            "model_annotator_count": a_counts,
            "random_accuracy": r_acc,
            "random_f1": r_f1,
            "random_precision": r_precision,
            "random_recall": r_recall,
        }
        scores.append(score)
        logger.log_metrics({"model_accuracy": acc}, step=alpha)
        logger.log_metrics({"model_precision": precision}, step=alpha)
        logger.log_metrics({"model_recall": recall}, step=alpha)
        logger.log_metrics({"model_f1": f1}, step=alpha)
        logger.log_metrics({"model_system_count": s_counts}, step=alpha)
        logger.log_metrics({"model_crowd_count": c_counts}, step=alpha)
        logger.log_metrics({"model_annotator_count": a_counts}, step=alpha)

        df = pd.DataFrame(data)
        # import ipdb;ipdb.set_trace()
        c_mat = confusion_matrix(df["answer"], df["model prediction"])
        tn = c_mat[0][0]
        fn = c_mat[1][0]
        tp = c_mat[1][1]
        fp = c_mat[0][1]

        logger.log_metrics({"alpha": alpha}, step=alpha)
        logger.log_metrics({"model true negative": tn}, step=alpha)
        logger.log_metrics({"model false negative": fn}, step=alpha)
        logger.log_metrics({"model true positive": tp}, step=alpha)
        logger.log_metrics({"model false positive": fp}, step=alpha)
        title = "result_model_{}_alpha_{}_seed_{}.csv".format(
            config.model, alpha, config.seed
        )
        df.to_csv("./output/" + title, index=False)
    scores = pd.DataFrame(scores)
    scores.to_csv("./output/scores.csv", index=False)


def calc_metrics(answer, result):
    acc = accuracy_score(answer, result)
    precision, recall, f1, _ = precision_recall_fscore_support(
        answer, result, average="binary"
    )
    return (acc, precision, recall, f1)


def eval_with_random(test, logger, config, alpha, c_count, a_count):
    crowd_d = test["crowd_dicision"].to_list()
    system_d = test["system_dicision"].to_list()
    answer = test["correct"].to_list()
    accs, precisions, recalls, f1s = [], [], [], []
    tns, tps, fns, fps, = (
        [],
        [],
        [],
        [],
    )
    # シード値かえて100かい回す
    for i in range(100):
        seed_everything(config.seed + i + 1)
        random_pred = RandomModel.predict(system_d, crowd_d, answer, c_count, a_count)
        acc = accuracy_score(answer, random_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            random_pred, answer, average="macro"
        )

        c_mat = confusion_matrix(answer, random_pred)
        tns.append(c_mat[0][0])
        fns.append(c_mat[1][0])
        tps.append(c_mat[1][1])
        fps.append(c_mat[0][1])
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    def calc_mean(l):
        return sum(l) / len(l)

    acc = calc_mean(accs)
    precision = calc_mean(precisions)
    recall = calc_mean(recalls)
    f1 = calc_mean(f1s)

    tn = calc_mean(tns)
    tp = calc_mean(tps)
    fn = calc_mean(fns)
    fp = calc_mean(fps)
    print(acc, precision, recall, f1)
    logger.log_metrics({"random_accuracy": acc}, step=alpha)
    logger.log_metrics({"random_precision": precision}, step=alpha)
    logger.log_metrics({"random_recall": recall}, step=alpha)
    logger.log_metrics({"random_f1": f1}, step=alpha)
    logger.log_metrics({"random true negative": tn}, step=alpha)
    logger.log_metrics({"random false negative": fn}, step=alpha)
    logger.log_metrics({"random true positive": tp}, step=alpha)
    logger.log_metrics({"random false positive": fp}, step=alpha)
    return (acc, f1, precision, recall)
