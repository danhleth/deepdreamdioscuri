import sys
from pathlib import Path
current_dir = str(Path(__file__).resolve().parent)
root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(root)

from dioscuri.base.opt import Opts
from dioscuri.sample_classification.pipeline import Pipeline

if __name__ == "__main__":
    opts = Opts().parse_args()
    train_pipeline = Pipeline(opts)
    train_pipeline.fit()