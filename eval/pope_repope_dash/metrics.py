# ash_eval/metrics.py

def pct(n, d):
    return f"{(n / d):.3%}  ({n}/{d})" if d else "n/a"


class BinaryYesNoMetrics:
    def __init__(self):
        self.n_total = 0
        self.acc_total = 0

        self.n_pos = 0
        self.acc_pos = 0

        self.n_neg = 0
        self.acc_neg = 0

    def update(self, pred_bin: int, gt_bin: int):
        correct = int(pred_bin == gt_bin)

        self.n_total += 1
        self.acc_total += correct

        if gt_bin == 1:
            self.n_pos += 1
            self.acc_pos += correct
        else:
            self.n_neg += 1
            self.acc_neg += correct

        return correct

    def print_report(self):
        fn = self.n_pos - self.acc_pos
        fp = self.n_neg - self.acc_neg

        print("\n================== Results ==================")
        print(f"Overall accuracy    : {pct(self.acc_total, self.n_total)}")
        print(f"POS (GT = yes)      : {pct(self.acc_pos, self.n_pos)}   <-- TP rate")
        print(f"NEG (GT = no)       : {pct(self.acc_neg, self.n_neg)}   <-- TN rate")
        print("--------------------------------------------")
        print(f"False Negative rate : {pct(fn, self.n_pos)}")
        print(f"False Positive rate : {pct(fp, self.n_neg)}")
        print("============================================")