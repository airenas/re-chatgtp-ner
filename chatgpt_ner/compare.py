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
    gt = pd.read_csv(args.gt, sep=',')
    # print(gt.head(10), sep='\n\n')
    pred = pd.read_csv(args.pred, sep=',')
    # print(pred.head(10), sep='\n\n')

    files = gt["file"].unique()
    logger.info("\nAuthors")
    all, ei, ed = 0, 0, 0
    for f in files:
        gt_f = gt[gt['file'] == f]
        pred_f = pred[pred['file'] == f]
        f_all, f_ei, f_ed = calc(gt_f, pred_f, "author")
        all, ei, ed = all + f_all, ei + f_ei, ed + f_ed
    logger.info("== all: {}\terr: {}\tins:{}\tdel: {}".format(all, ei + ed, ei, ed))

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
    logger.info("== all: {}\terr: {}\tins:{}\tdel: {}".format(all, ei + ed, ei, ed))

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
    logger.info("== all: {}\terr: {}\tins:{}\tdel: {}".format(all, ei + ed, ei, ed))

    # y_true = entries["RecAccount"].values.tolist()
    # y_rec_type = [e_str(v) for v in entries["RecType"].values.tolist()]
    #
    # y_true = ["%s:%s" % (y_rec_type[i], v) for i, v in enumerate(y_true)]
    # y_true_docs = entries["RecDocs"].values.tolist()
    # y_true_docs = [e_str(x) for x in y_true_docs]
    #
    # skip = args.skip
    # y_true = y_true[skip:len(y_pred_l)]
    # y_true_docs = y_true_docs[skip:len(y_pred_l)]
    # y_pred = [y.split('\t')[0] for y in y_pred_l[skip:]]
    # y_pred_docs = [y.split('\t')[1] for y in y_pred_l[skip:]]
    # y_pred_v = [json.loads(y.split('\t')[2]) for y in y_pred_l[skip:]]
    #
    # y_pred_reject = [x if sim_val(y_pred_v[i]) > args.limit else 'rejected' for i, x in enumerate(y_pred)]
    #
    # with open(args.out, 'w') as f:
    #     for i, y in enumerate(y_true):
    #         ir = i + skip
    #         v = y_pred_reject[i]
    #         vec = y_pred_v[i]
    #         val = sim_val(vec)
    #         if v == y and val > args.limit:
    #             print("{}\t{}\t{}".format(ir, y, '' if y_rec_type[i] else 'empty'), file=f)
    #         else:
    #             print("{}\t{}\t{}\t{} <--diff-->\t{}\t{}".format(ir, y, v, '' if y_rec_type[i] else 'empty',
    #                                                              vec, val), file=f)
    #
    # y_true_n = [x for i, x in enumerate(y_true) if y_rec_type[i] != '']
    # y_pred_n = [x for i, x in enumerate(y_pred) if y_rec_type[i] != '']
    #
    # logger.info("Empty              : {:.3f} ({}/{})".format((len(y_true) - len(y_true_n)) / len(y_true), len(y_true_n),
    #                                                          len(y_true)))
    # logger.info("Acc all            : {:.3f} ({}/{})".format(accuracy_score(y_true_n, y_pred_n),
    #                                                          sum([1 for i, x in enumerate(y_true_n) if
    #                                                               y_pred_n[i] != x]),
    #                                                          len(y_true_n)))
    # y_pred_nreject = [x for i, x in enumerate(y_pred_reject) if y_rec_type[i] != '']
    # show_no_rejected(y_true_n, y_pred_nreject, 'Acc not rejected   ', lambda p: True)
    # show_no_rejected(y_true_n, y_pred_nreject, 'Acc BA             ', lambda p: p.startswith(LType.BA.to_s()))
    # show_no_rejected(y_true_n, y_pred_nreject, 'Acc GL             ', lambda p: p.startswith(LType.GL.to_s()))
    # show_no_rejected(y_true_n, y_pred_nreject, 'Acc Customer       ', lambda p: p.startswith(LType.CUST.to_s()))
    # show_no_rejected(y_true_n, y_pred_nreject, 'Acc Vendor         ', lambda p: p.startswith(LType.VEND.to_s()))
    #
    # logger.info("Docs ...")
    # rda, rds, rdi, rdd, rs, ny = 0, 0, 0, 0, 0, 0
    # with open(args.out + '.docs', 'w') as f:
    #     for i, y in enumerate(y_true_docs):
    #         if not y:
    #             ny += 1
    #             continue
    #         ir = i + skip
    #         pa = y_pred_docs[i].split(";")
    #         ya = y.split(";")
    #         vec = y_pred_v[i]
    #         val = sim_val(vec)
    #         se, ie, de, r = 0, 0, 0, ""
    #         if val > args.limit:
    #             a, se, ie, de = cmp_arr(ya, pa)
    #             rda, rds, rdi, rdd = rda + a, rds + se, rdi + ie, rdd + de
    #         else:
    #             r = "rejected"
    #             rs += len(ya)
    #         if (se + ie + de) == 0 and val > args.limit:
    #             print("{}\t{}".format(ir, y), file=f)
    #         else:
    #             print("{}\t{} {}<--diff-->\t{}".format(ir, y, r, y_pred_docs[i]), file=f)
    #
    # logger.info(
    #     "Acc all {:.3f} ({}/{}) s:{}, i:{}, d:{}\t(rejected {}, no doc: {})".format(1 - ((rds + rdd + rdi) / rda),
    #                                                                                 (rds + rdd + rdi), rda, rds,
    #                                                                                 rdi, rdd, rs, ny))
    logger.info("done")


if __name__ == "__main__":
    main(sys.argv[1:])
