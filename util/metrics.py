import torch
from torch import distributed as dist
from util.misc import bb_intersection_over_union

def all_reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)

    return rt


class NounACC:
    def __init__(self, top_k, distributed=True):
        self.top_k = top_k
        self.distributed = distributed
        self.batch_results = []

    def update(self,
               gt_verb,
               gt_noun1,
               gt_noun2,
               gt_noun3,
               pred_verb=None,
               pred_noun=None,
               pred_boxes=None,
               gt_boxes=None):
        """
        :param gt_verb: [n]
        :param gt_noun1: [n, 6]
        :param gt_noun2: [n, 6]
        :param gt_noun3: [n, 6]
        :param pred_verb: [n, verb_num]
        :param pred_noun: [n, noun_num, 6]
        :param pred_box: [n, 6, 4]
        """
        batch_size = pred_noun.size(0)
        role_num = torch.where(gt_noun1 == -1,
                               torch.zeros_like(gt_noun1),
                               torch.ones_like(gt_noun1))
        role_num = torch.sum(role_num, dim=1)  # [n]
        batch_result = pred_noun.new_zeros((11))  # [verb, value, value*, value-all, value-all*, grnd-value, grnd-value*, grnd-value-all, grnd-value-all*, num_img, num_value]

        batch_result[9] = batch_size
        batch_result[10] = torch.sum(role_num)

        for i in range(batch_size):
            gt_v = gt_verb[i]  # [1]
            if pred_verb is not None:
                pred_v = pred_verb[i]  # [verb_num]
                if len(pred_verb.shape) == 1:
                    verb_found = pred_v == gt_v
                else:
                    verb_found = (gt_v in pred_v)
            else:
                verb_found = True

            if verb_found:
                batch_result[0] += 1

            gt_role_count = role_num[i]

            all_found = True
            all_match = True
            for k in range(gt_role_count):
                if len(pred_noun.size()) == 3:
                    pred_id = torch.max(pred_noun[i, :, k], 0)[1]
                else:
                    pred_id = pred_noun[i,k]
                pred_box = pred_boxes[i][k]
                gt_box = gt_boxes[i][k]
                box_match = bb_intersection_over_union(pred_box, gt_box)

                found = False

                gt_n1 = gt_noun1[i][k]
                gt_n2 = gt_noun2[i][k]
                gt_n3 = gt_noun3[i][k]
                if pred_id == gt_n1:
                    found = True
                if pred_id == gt_n2:
                    found = True
                if pred_id == gt_n3:
                    found = True
                if not box_match:
                    all_match = False
                if not found:
                    all_found = False
                if found and verb_found:
                    batch_result[1] += 1
                    if box_match:
                        batch_result[5] += 1
                if found:
                    batch_result[2] += 1
                    if box_match:
                        batch_result[6] += 1

            if all_found and verb_found:
                batch_result[3] += 1
                if all_match:
                    batch_result[7] += 1

            if all_found:
                batch_result[4] += 1
                if all_match:
                    batch_result[8] += 1

        if self.distributed:
            batch_result = all_reduce_sum(batch_result)

        self.batch_results.append(batch_result.unsqueeze(0))

    def get_value(self):
        res = torch.cat(self.batch_results, dim=0)
        res = torch.sum(res, dim=0)  # [7]
        dict_res = {'verb': res[0] / res[9],
                    'value': res[1] / res[10],
                    'value*': res[2] / res[10],
                    'value-all': res[3] / res[9],
                    'value-all*': res[4] / res[9],
                    'grnd-value': res[5] / res[10],
                    'grnd-value*': res[6] / res[10],
                    'grnd-value-all': res[7] / res[9],
                    'grnd-value-all*': res[8] / res[9]
                    }

        return dict_res
