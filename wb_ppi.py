import warnings

warnings.filterwarnings("ignore")

from strategies import strategies
import atlases

import argparse
from load_confounds import Confounds
from bids.layout import parse_file_entities
from pathlib import Path
import numpy as np
import pandas as pd

from nilearn.input_data import NiftiLabelsMasker
from nilearn.glm.first_level import make_first_level_design_matrix
from sklearn.linear_model import LinearRegression


def get_args():
    parser = argparse.ArgumentParser(description="whole-brain PPI")
    parser.add_argument("fmriprep", type=Path, help="path to fmriprep directory")
    parser.add_argument("tr", type=float, help="repetition time in seconds")
    parser.add_argument("strategy", type=str, help=" - ".join(strategies.keys()))
    parser.add_argument("atlas", type=str, help=" - ".join(atlases.atlas_options))
    parser.add_argument(
        "--smooth_fwhm",
        type=float,
        default=6,
        help="smoothing kernel, default fwhm 6mm",
    )
    args = parser.parse_args()
    assert (
        args.strategy in strategies.keys()
    ), f"{args.strategy} is not a valid strategy"
    assert args.atlas in atlases.atlas_options, f"{args.atlas} not an available atlas"
    assert args.fmriprep.is_dir(), "fmriprep directory does not exist"
    return args


class Data:
    def __init__(self, fmriprep, strategy):
        print("\nindexing files... ", end="")
        self.fmriprep = fmriprep
        self.preprocs = self.get_preprocs()
        self.events = self.get_events()
        self.confounds = self.get_confounds(strategy)
        assert len(self.preprocs) == len(self.confounds), "missings fmriprep files"
        print(f"found {len(self.preprocs)} images")

    def get_preprocs(self):
        preprocs = self.fmriprep.glob(
            "**/sub-*_task-h*_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
        )
        return sorted([str(preproc) for preproc in preprocs])

    def get_events(self):
        events = sorted(
            self.fmriprep.parents[1].glob("rawdata/**/sub-*_task-h*events.tsv")
        )
        return [
            pd.read_csv(event, sep="\t", usecols=["onset", "duration", "trial_type"])
            for event in events
        ]

    def get_confounds(self, strategy):
        return Confounds(**strategy).load(self.preprocs)


def get_linear_model():
    return LinearRegression(fit_intercept=False, n_jobs=-2)


def build_path(derivatives, sub, task, strategy, atlas):
    path = derivatives / f"wb_ppi/sub-{sub}"
    path.mkdir(parents=True, exist_ok=True)
    # BIDS pattern: sub-{subject}_task-{task}_atlas-{atlas}_strat-{strategy}_{suffix}.{extension}
    return path / f"sub-{sub}_task-{task}_atlas-{atlas}_strat-{strategy}_ppi.npy"


def save_df(path, atlas):
    atlas.df.to_csv(f"{path}/atlas-{atlas.title}_labels.csv", index=False)


def save_fig(path, atlas):
    atlas.fig.savefig(f"{path}/atlas-{atlas.title}_fig.png", dpi=300)


def save_record(path, args):
    file = path / "record.csv"
    try:
        record = pd.read_csv(file)
    except:
        record = pd.DataFrame({"atlas": [], "strategy": [], "smooth_fwhm": []})
    record.loc[len(record)] = [args.atlas, args.strategy, args.smooth_fwhm]
    record.to_csv(file, index=False)


def get_dm(event, frame_times):
    return make_first_level_design_matrix(
        frame_times=frame_times,
        events=event,
        hrf_model="glover + derivative",
        drift_model=None,
        high_pass=None,
    )


def fit_ppi(ts_rois, dm, model):

    ppi_vect = []

    for roi in ts_rois.T:
        dm["roi"] = roi  # is mean centered
        psy = dm.iloc[:, 0]
        psy = psy - np.mean([psy.min(), psy.max()])  # zero center
        dm["ppi"] = psy * dm["roi"]
        model.fit(dm, ts_rois)  # y = all rois, X = dm
        ppi_vect.append(
            model.coef_[:, -1]
        )  # beta coefficients corresponding to ppi regressor

    return np.hstack(ppi_vect)


def main():
    args = get_args()
    derivatives = args.fmriprep.parent
    data = Data(args.fmriprep, strategies[args.strategy])
    atlas = atlases.Atlas(args.atlas)
    masker = atlas.get_masker(
        atlas.maps, atlas.probabilistic, derivatives, args.smooth_fwhm
    )
    model = get_linear_model()

    for preproc, event, confound in zip(data.preprocs, data.events, data.confounds):

        file_entities = parse_file_entities(preproc)
        sub, task = file_entities["subject"], file_entities["task"]
        print(f"> sub-{sub} task-{task}")

        path_ppi = build_path(derivatives, sub, task, args.strategy, atlas.title)

        ts_rois = masker.fit_transform(imgs=preproc, confounds=confound)
        samples = len(ts_rois)

        frame_times = np.linspace(0, (samples * args.tr) - args.tr, samples)

        dm = get_dm(event, frame_times)

        ppi_vect = fit_ppi(ts_rois, dm, model)
        np.save(path_ppi, ppi_vect)

    path = derivatives / f"wb_ppi/group/atlas-{atlas.title}"
    path.mkdir(parents=True, exist_ok=True)

    save_df(path, atlas)
    save_fig(path, atlas)
    save_record(path.parent, args)


if __name__ == "__main__":
    main()
