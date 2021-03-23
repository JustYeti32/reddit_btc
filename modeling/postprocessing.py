import matplotlib.pyplot as plt


def aggregate_signals(raw_signals, pace="1D", plot=True):
    signals = raw_signals.resample(pace, label="right", closed="right").mean().rename("signals").dropna()
    signals = signals.apply(round)
    supports = raw_signals.resample(pace, label="right", closed="right").count().rename("support")

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(18, 6))
        signals.plot(ax=ax[0], label="signals")
        supports.plot(ax=ax[1], label="support")

        ax[0].legend(loc="upper left")
        ax[1].legend(loc="upper left")

    return signals, supports

########################################################################################################################