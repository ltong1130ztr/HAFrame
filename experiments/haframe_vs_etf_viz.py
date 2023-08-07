import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--etf-path", type=str)
parser.add_argument("--haframe-path", type=str)
parser.add_argument("--partition", type=str, default="train")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--ckpt-freq", type=int, default=5)
parser.add_argument("--output", type=str)
nc_opts = parser.parse_args()


def main(opts):

    assert opts.partition in opts.etf_path
    assert opts.partition in opts.haframe_path
    assert opts.epochs > opts.ckpt_freq

    # load etf and haframe collpase records
    with open(opts.etf_path, "rb") as f:
        etf_data = pickle.load(f)

    cls_cos_mean_etf = etf_data['cls_cos_mean']
    cls_cos_std_etf = etf_data['cls_cos_std']
    cfeat_cos_mean_etf = etf_data['cfeat_cos_mean']
    cfeat_cos_std_etf = etf_data['cfeat_cos_std']
    cls_cfeat_duality = etf_data['cls_cfeat_duality']

    with open(opts.haframe_path, "rb") as f:
        haframe_data = pickle.load(f)

    cls_cos_mean_haf = haframe_data['cls_cos_mean']
    cls_cos_std_haf = haframe_data['cls_cos_std']
    rfeat_cos_mean_haf = haframe_data['rfeat_cos_mean']
    rfeat_cos_std_haf = haframe_data['rfeat_cos_std']
    cls_rfeat_duality = haframe_data['cls_rfeat_duality']

    # output viz results
    fig1_path = os.path.join(opts.output, f"{opts.partition}_pfeature_nc_std.jpg")
    fig2_path = os.path.join(opts.output, f"{opts.partition}_pfeature_nc_mean.jpg")
    fig3_path = os.path.join(opts.output, f"{opts.partition}_pfeature_self_duality.jpg")

    epochs = np.arange(opts.ckpt_freq, opts.epochs, opts.ckpt_freq)

    # std viz
    fig1 = plt.figure(figsize=(10, 8))
    plt.plot(epochs, cfeat_cos_std_etf, ">-", color="salmon", markersize=15, label="centered feature (ETF)")
    plt.plot(epochs, cls_cos_std_etf, "*-", color="red", markersize=15, label="learnable classifiers (ETF)")
    plt.plot(epochs, rfeat_cos_std_haf, "x-", color="royalblue", markersize=15, label="transformed feature (HAF)")
    plt.plot(epochs, cls_cos_std_haf, "o-", color="blue", markersize=15, label="fixed classifiers (HAF)")
    plt.xlabel("epochs", fontsize=25)
    plt.ylabel(r"$std(cos)$", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylim((0, 0.3))
    plt.legend(fontsize=20)
    plt.grid(True)
    fig1.savefig(fig1_path)
    print(f"saving figure at {fig1_path}")

    # mean viz
    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(epochs, cfeat_cos_mean_etf, ">-", color="salmon", markersize=15, label="centered feature (ETF)")
    plt.plot(epochs, cls_cos_mean_etf, "*-", color="red", markersize=15, label="learnable classifiers (ETF)")
    plt.plot(epochs, rfeat_cos_mean_haf, "x-", color="royalblue", markersize=15, label="transformed feature (HAF)")
    plt.plot(epochs, cls_cos_mean_haf, "o-", color="blue", markersize=15, label="fixed classifiers (HAF)")
    plt.xlabel("epochs", fontsize=25)
    plt.ylabel(r"$mean(cos)$", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylim((0, 0.8))
    plt.legend(fontsize=20)
    plt.grid(True)
    fig2.savefig(fig2_path)
    print(f"saving figure at {fig2_path}")

    # self-duality viz
    fig3 = plt.figure(figsize=(10, 8))
    plt.plot(epochs, cls_cfeat_duality, "*-", color='red', markersize=15, label="ETF")
    plt.plot(epochs, cls_rfeat_duality, ">-", color='blue', markersize=15, label="HAFrame")
    plt.xlabel("epochs", fontsize=25)
    plt.ylabel(r"$||W-H||$", fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylim((0, 0.8))
    plt.legend(fontsize=20)
    plt.grid(True)
    fig3.savefig(fig3_path)
    print(f"saving figure at {fig3_path}")


if __name__ == "__main__":
    main(nc_opts)
