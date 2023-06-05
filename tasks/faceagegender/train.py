import sys
from pathlib import Path
root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(root)

import matplotlib as mpl

mpl.use('Agg')
from dioscuri.base.opt import Opts
from dioscuri.faceagegender.pipeline import Pipeline

if __name__ == "__main__":
    opts = Opts.parse("./configs/opt.yaml")
    train_pipeline = Pipeline(opts)
    train_pipeline.fit()