import pandas as pd
from nilearn import datasets, input_data, plotting

atlas_options = ["yeo2011", "schaefer2018", "difumo2020", "ajd2021", "harvox2006"]


class Atlas:
    def __init__(self, atlas):

        self.title = atlas
        self.maps, self.df, self.probabilistic = self.get_data()
        self.df[["x", "y", "z"]] = self.get_coords()
        self.fig = self.get_fig()

    def get_data(self):

        if self.title == "yeo2011":
            fetcher = datasets.fetch_atlas_yeo_2011()
            maps = fetcher.thick_7
            df = pd.read_csv("atlases/atlas-yeo2011.csv")
            probabilistic = False

        elif self.title == "schaefer2018":
            fetcher = datasets.fetch_atlas_schaefer_2018(
                n_rois=100, yeo_networks=7, resolution_mm=2
            )
            maps = fetcher.maps
            df = [label.decode() for label in fetcher.labels]
            df = pd.DataFrame(df, columns=["labels"])
            probabilistic = False

        elif self.title == "difumo2020":
            fetcher = datasets.fetch_atlas_difumo(dimension=64)
            maps = fetcher.maps
            df = pd.DataFrame(
                [label[1] for label in fetcher.labels], columns=["labels"]
            )
            probabilistic = True

        elif self.title == "ajd2021":
            maps = "atlases/atlas-ajd2021.nii.gz"
            df = pd.read_csv("atlases/atlas-ajd2021.csv")
            probabilistic = False

        elif self.title == "harvox2006":
            fetcher = datasets.fetch_atlas_harvard_oxford(
                "cort-maxprob-thr25-2mm", symmetric_split=True
            )
            maps = fetcher.maps
            df = pd.DataFrame(fetcher.labels[1:], columns=["labels"])
            probabilistic = False

        return maps, df, probabilistic

    def get_coords(self):

        if self.probabilistic:
            return plotting.find_probabilistic_atlas_cut_coords(maps_img=self.maps)

        else:
            return plotting.find_parcellation_cut_coords(labels_img=self.maps)

    def get_fig(self):

        kwargs = {"display_mode": "z", "annotate": False, "draw_cross": False}

        if self.probabilistic:
            return plotting.plot_prob_atlas(self.maps, **kwargs)

        else:
            return plotting.plot_roi(self.maps, **kwargs)

    def get_masker(self, maps, probabilistic, derivatives, smooth_fwhm):

        kwargs = {
            "smoothing_fwhm": smooth_fwhm,
            "standardize": True,
            "standardize_confounds": True,
            "memory": str(derivatives / ".wb_ppi_cache"),
            "memory_level": 3,
        }

        if probabilistic:
            return input_data.NiftiMapsMasker(maps_img=maps, **kwargs)

        else:
            return input_data.NiftiLabelsMasker(
                labels_img=maps, strategy="mean", **kwargs
            )
