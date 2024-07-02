from PYEVALB import scorer as scorer_module
from PYEVALB import parser
from PYEVALB.summary import Result, summary
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
from src.utils.constituency_parsing_utils import remove_space

gold = "(IP (NP (PN 这里)) (VP (ADVP (AD 便)) (VP (VV 产生) (IP (NP (QP (CD 一) (CLP (M 个))) (DNP (NP (JJ 结构性)) (DEG 的)) (NP (NN 盲点))) (PU ：) (IP (VP (VV 臭味相投) (PU ，) (VV 物以类聚)))))) (PU 。))"

test = "(IP (IP (NP (PN 这里)) (VP (ADVP (AD 便)) (VP (VV 产生) (NP (QP (CD 一) (CLP (M 个))) (DNP (ADJP (JJ 结构性)) (DEG 的)) (NP (NN 盲点)))))) (PU ：) (IP (NP (NN 臭味相投)) (PU ，) (VP (VV 物以类聚))) (PU 。))"

gold_tree = parser.create_from_bracket_string(gold)
test_tree = parser.create_from_bracket_string(test)
scorer = scorer_module.Scorer()
result = scorer.score_trees(gold_tree, test_tree)

print("Recall =" + str(result.recall))
print("Precision =" + str(result.prec))

for stat in result.STATISTICS_TABLE:
    print(stat + " = " + str(getattr(result, stat)))


# check if all
from datasets import load_dataset

dataset = load_dataset("saibo/ptb-test-1k-llm-few-shot")["llama2_70b"]
results = []
for i, row in enumerate(dataset):
    if i in [730, 770, 783, 118, 154, 168, 221, 252, 392, 423, 623, 751, 877, 911]:
        continue
    # print(f"Processing row {i}")
    # string = replace_words_with_xxx(remove_space(row["target"]))
    string = row["target"]
    no_space_string = remove_space(string) + ")"
    # score
    # print(string)
    gold_tree = parser.create_from_bracket_string(string)
    test_tree = parser.create_from_bracket_string(no_space_string)
    result = scorer.score_trees(gold_tree, test_tree)
    # print('Recall =' + str(result.recall))
    # print('Precision =' + str(result.prec))
    if result.state == 1:
        raise ValueError("Error while scoring the trees")
    if result.recall < 1 or result.prec < 1:
        # 783
        # raise ValueError(f"Recall or Precision is less than 1 for row {i}")
        print(f"Recall or Precision is less than 1 for row {i}")
    results.append(result)

result_summary = summary(results)
print(result_summary)
