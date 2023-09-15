import argparse
import sys

import pandas as pd

from chatgpt_ner.logger import logger


def get(gt, param):
    res = gt.get(param, "")
    if res != str(res):
        return ""
    return res


def calc(gt, pred_f, param):
    wanted = get(gt, param)
    got = get(pred_f, param)
    if wanted != got:
        logger.debug(f"{param}: '{wanted}' vs '{got}'")
        return 1
    return 0


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
    gt = pd.read_csv(args.gt, sep=',', dtype=str).to_dict('records')
    pred = pd.read_csv(args.pred, sep=',', dtype=str).to_dict('records')
    gtd = {d.get("institution"): d for d in gt}
    predd = {d.get("name"): d for d in pred}

    institutions = gtd.keys()
    test_fiels = ["institutionName", "addressLine", "city", "state", "postCode", "country"]
    for f in test_fiels:
        logger.info(f"\n{f}")
        all, es = 0, 0
        for inst in institutions:
            gt_f = gtd[inst]
            pred_f = predd.get(inst, None)
            if pred_f is None:
                logger.info("No data for '{}'".format(inst))
                continue
            f_es = calc(gt_f, pred_f, f)
            all, es = all + 1, es + f_es
        logger.info("== all: {}\terr: {}".format(all, es))
    logger.info("done")


if __name__ == "__main__":
    main(sys.argv[1:])
