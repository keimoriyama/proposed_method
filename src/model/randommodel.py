import random

from model.modelinterface import ModelInterface


class RandomModel(ModelInterface):
    def __init__(self, out_dim) -> None:
        super().__init__()
        self.out_dim = out_dim

    @classmethod
    def predict(
        cls, system_dicision, crowd_dicision, anotator, crowd_count, annotator_count
    ):
        model_ans = []
        indexes = [i for i in range(len(system_dicision))]
        crowd_i = random.sample(indexes, crowd_count)
        for c_i in crowd_i:
            indexes.remove(c_i)
        ann_i = random.sample(indexes, annotator_count)
        crowd_i = set(crowd_i)
        ann_i = set(ann_i)
        c_counts, a_counts = 0, 0
        for i in range(len(crowd_dicision)):
            if i in crowd_i:
                model_ans.append(crowd_dicision[i])
                c_counts += 1
            elif i in ann_i:
                model_ans.append(anotator[i])
                a_counts += 1
            else:
                model_ans.append(system_dicision[i])
        assert c_counts == crowd_count
        assert a_counts == annotator_count
        return model_ans
