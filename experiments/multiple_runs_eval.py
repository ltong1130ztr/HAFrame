"""
load multiple runs of test_summary.json/validation_summary.json files
compute average and 95% confidence interval
"""
import os
import json
import argparse
import numpy as np
from scipy import stats


def mean_and_conf_interval_z_score(opts):
    summaries = dict()
    z_value = 1.96  # for two-tailed 95% confidence interval
    for s in opts.seeds:
        if opts.postfix_tag:
            summary_path = os.path.join(opts.output, f"{opts.loss}-{opts.arch}-{opts.postfix_tag}-seed_{s}",
                                        f"{opts.partition}_summary.json")
        else:
            summary_path = os.path.join(opts.output, f"{opts.loss}-{opts.arch}-seed_{s}",
                                        f"{opts.partition}_summary.json")
        with open(summary_path, "r") as fp:
            summary = json.load(fp)
            print(f"load summary: {summary_path}")
        for k in summary.keys():
            val = summary[k]
            if k not in summaries:
                summaries[k] = []
            summaries[k].append(val)

    for k in summaries.keys():
        avg = np.mean(summaries[k])
        conf95 = z_value * np.std(summaries[k]) / np.sqrt(len(summaries[k]))
        print("\t\t\t\t%20s: %.2f" % (k, avg) + " +/- %.4f" % conf95)

    return


def mean_and_conf_interval_z_score_crm(opts):
    summaries = dict()
    z_value = 1.96  # for two-tailed 95% confidence interval
    for s in opts.seeds:
        if opts.postfix_tag:
            summary_path = os.path.join(opts.output, f"{opts.loss}-{opts.arch}-{opts.postfix_tag}-seed_{s}",
                                        f"base_n_crm_{opts.partition}_summary.json")
        else:
            summary_path = os.path.join(opts.output, f"{opts.loss}-{opts.arch}-seed_{s}",
                                        f"base_n_crm_{opts.partition}_summary.json")
        with open(summary_path, "r") as fp:
            summary = json.load(fp)
            print(f"load summary: {summary_path}")
        for k in summary.keys():
            if k in ["base_res_str", "crm_res_str"]: continue
            val = summary[k]
            if k not in summaries:
                summaries[k] = []
            summaries[k].append(val)

    for k in summaries.keys():
        avg = np.mean(summaries[k])
        conf95 = z_value * np.std(summaries[k]) / np.sqrt(len(summaries[k]))
        print("\t\t\t\t%20s: %.2f" % (k, avg) + " +/- %.4f" % conf95)

    return


def mean_and_conf_interval_t_score(opts):
    print(f"One sample confidence interval from t-distribution")
    summaries = dict()
    alpha = 0.05
    dof = len(opts.seeds) - 1
    # t-value for two-tailed 95% confidence interval with degree of freedom = n - 1
    t_value = stats.t.isf(alpha / 2, dof)
    for s in opts.seeds:
        if opts.postfix_tag:
            summary_path = os.path.join(opts.output, f"{opts.loss}-{opts.arch}-{opts.postfix_tag}-seed_{s}",
                                        f"{opts.partition}_summary.json")
        else:
            summary_path = os.path.join(opts.output, f"{opts.loss}-{opts.arch}-seed_{s}",
                                        f"{opts.partition}_summary.json")
        with open(summary_path, "r") as fp:
            summary = json.load(fp)
            print(f"load summary: {summary_path}")
        for k in summary.keys():
            val = summary[k]
            if k not in summaries:
                summaries[k] = []
            summaries[k].append(val)

    for k in summaries.keys():
        avg = np.mean(summaries[k])
        conf95 = t_value * np.std(summaries[k]) / np.sqrt(len(summaries[k]))
        print("\t\t\t\t%20s: %.2f" % (k, avg) + " +/- %.4f" % conf95)

    return


def mean_and_conf_interval_t_score_crm(opts):
    print(f"One sample confidence interval from t-distribution")
    summaries = dict()
    alpha = 0.05
    dof = len(opts.seeds) - 1
    # t-value for two-tailed 95% confidence interval with degree of freedom = n - 1
    t_value = stats.t.isf(alpha / 2, dof)
    for s in opts.seeds:
        if opts.postfix_tag:
            summary_path = os.path.join(opts.output, f"{opts.loss}-{opts.arch}-{opts.postfix_tag}-seed_{s}",
                                        f"base_n_crm_{opts.partition}_summary.json")
        else:
            summary_path = os.path.join(opts.output, f"{opts.loss}-{opts.arch}-seed_{s}",
                                        f"base_n_crm_{opts.partition}_summary.json")
        with open(summary_path, "r") as fp:
            summary = json.load(fp)
            print(f"load summary: {summary_path}")
        for k in summary.keys():
            if k in ["base_res_str", "crm_res_str"]: continue
            val = summary[k]

            if k not in summaries:
                summaries[k] = []
            summaries[k].append(val)
    for k in summaries.keys():
        avg = np.mean(summaries[k])
        conf95 = t_value * np.std(summaries[k]) / np.sqrt(len(summaries[k]))
        print("\t\t\t\t%20s: %.2f" % (k, avg) + " +/- %.4f" % conf95)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--postfix-tag", type=str, default=None)
    parser.add_argument("--partition", type=str, default="test")
    parser.add_argument("--nseed", type=int, default=5)
    parser.add_argument("--crm", action="store_true")
    parser.add_argument("--score", type=str, default="t", choices=["t", "z"])  # t-score or z-score

    eval_opts = parser.parse_args()
    eval_opts.seeds = [i for i in range(0, eval_opts.nseed)]

    if not eval_opts.crm:
        if eval_opts.score == "z":
            mean_and_conf_interval_z_score(eval_opts)
        else:
            mean_and_conf_interval_t_score(eval_opts)
    else:
        if eval_opts.score == "z":
            mean_and_conf_interval_z_score_crm(eval_opts)
        else:
            mean_and_conf_interval_t_score_crm(eval_opts)

