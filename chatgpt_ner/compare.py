import argparse
import sys

import pandas as pd

from chatgpt_ner.logger import logger


def calc(gt, pred_f, param):
    wanted = gt[param].dropna().unique()
    got = pred_f[param].dropna().unique()
    ok = 0
    for item1 in wanted:
        if item1 in got:
            ok += 1
        else:
            logger.debug(f"No item: {item1}")
    d = len(wanted) - ok
    i = max(len(got) - ok, 0)
    if i > 0:
        for item1 in got:
            if item1 not in wanted:
                logger.debug(f"Inserted: {item1}")
    return len(wanted), i, d


def err_p(total, errs):
    if total == 0:
        return 0
    return errs / total * 100


def main(argv):
    parser = argparse.ArgumentParser(description="Compares two files",
                                     epilog="E.g. " + sys.argv[0] + "",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gt", nargs='?', required=True, help="File 1")
    parser.add_argument("--pred", nargs='?', required=True, help="File 2")
    args = parser.parse_args(args=argv)

    logger.info("Starting")
    logger.info("GT         : {}".format(args.gt))
    logger.info("Prediction : {}".format(args.pred))
    gt = pd.read_csv(args.gt, sep=',', dtype=str)
    # print(gt.head(10), sep='\n\n')
    pred = pd.read_csv(args.pred, sep=',', dtype=str)
    # print(pred.head(10), sep='\n\n')

    files = gt["file"].unique()
    logger.info("\nAuthors")
    all, ei, ed = 0, 0, 0
    for f in files:
        gt_f = gt[gt['file'] == f]
        pred_f = pred[pred['file'] == f]
        f_all, f_ei, f_ed = calc(gt_f, pred_f, "author")
        all, ei, ed = all + f_all, ei + f_ei, ed + f_ed
    logger.info("== all: {}\terr: {:.2f}% {}\tins:{}\tdel: {}".format(all, err_p(all, ei + ed), ei + ed, ei, ed))

    logger.info("\nAuthors-institutions")
    all, ei, ed = 0, 0, 0
    for f in files:
        gt_f = gt[gt['file'] == f]
        pred_f = pred[pred['file'] == f]
        authors = gt["author"].unique()
        for a in authors:
            gt_a = gt_f[gt_f['author'] == a]
            pred_a = pred_f[pred_f['author'] == a]
            f_all, f_ei, f_ed = calc(gt_a, pred_a, "institution")
            all, ei, ed = all + f_all, ei + f_ei, ed + f_ed
    logger.info("== all: {}\terr: {:.2f}% {} \tins:{}\tdel: {}".format(all, err_p(all, ei + ed), ei + ed, ei, ed))

    logger.info("\nAuthors-emails")
    all, ei, ed = 0, 0, 0
    for f in files:
        gt_f = gt[gt['file'] == f]
        pred_f = pred[pred['file'] == f]
        authors = gt["author"].unique()
        for a in authors:
            gt_a = gt_f[gt_f['author'] == a]
            pred_a = pred_f[pred_f['author'] == a]
            f_all, f_ei, f_ed = calc(gt_a, pred_a, "email")
            all, ei, ed = all + f_all, ei + f_ei, ed + f_ed
    logger.info("== all: {}\terr: {:.2f}% {}\tins:{}\tdel: {}".format(all, err_p(all, ei + ed), ei + ed, ei, ed))
    logger.info("done")


if __name__ == "__main__":
    main(sys.argv[1:])
