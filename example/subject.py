import warnings

warnings.filterwarnings("ignore")
from bids import BIDSLayout
from pathlib import Path
import numpy as np
import pandas as pd

path = Path("/Volumes/MRI/mfc")
wb_ppi = path / "derivatives/wb_ppi"
subjects = BIDSLayout(
    root=path / "rawdata", validate=False, index_metadata=False
).get_subjects()
record = pd.read_csv(wb_ppi / "group/record.csv")

for row in record.values:
    print(row)

    for sub in subjects:
        files = sorted(
            wb_ppi.glob(
                f"sub-{sub}/sub-{sub}_task-h?_atlas-{row[0]}_strat-{row[1]}_ppi.npy"
            )
        )
        ppi_vect = np.array([np.load(file) for file in files])

        pre, post = ppi_vect[:2,].mean(axis=0), ppi_vect[
            2:,
        ].mean(axis=0)
        mean = np.mean([pre, post], axis=0)
        diff = post - pre

        template = str(files[0])
        for i in [("pre", pre), ("post", post), ("mean", mean), ("diff", diff)]:
            np.save(template.replace("task-h1", f"manip-{i[0]}"), i[1])
