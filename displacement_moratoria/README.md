# Strategic displacement and moratorium contagion on jurisdictional networks

Code and data for the computations in the manuscript *Strategic Displacement and Moratorium Contagion on Jurisdictional Networks* (Moses Boudourides).

## Contents

| Path | Description |
|---|---|
| `paper_computations.py` | Synthetic experiments: checks of the exact statements, exposure along simulated cascades (Table 1, Figure 1), confinement and crossing with two types (Table 2), complementarity with spillover costs |
| `michigan_illustration.py` | Michigan illustration: adjacency graph of county subdivisions, displacement graph, exposure and adoption assortativity over time (Table 3, Figure 2) |
| `data/mi_moratoria.csv` | Moratoria compiled from the Siting Radar website, WKAR, datacenterbans.com, the Baldori Law tracker and local sources |
| `data_raw/mi_cousub_2020_500k.json` | U.S. Census Bureau 2020 cartographic boundary file, Michigan county subdivisions, 1:500,000 (public domain), from the Census Bureau's [citysdk](https://github.com/uscensusbureau/citysdk) repository |
| `data_raw/sr/` | Place for the Siting Radar export (not redistributed; see below) |
| `results/`, `tables/`, `figures/` | Outputs of the two scripts |

## Siting Radar data

The Michigan illustration also uses the Siting Radar export of detected local actions on data-centre moratoria. It is not included here. Download it from <https://sitingradar.com/data-center-moratoria.csv> and save it as

```
data_raw/sr/siting_radar_data_center_moratoria.csv
```

The version used in the manuscript was downloaded on 8 November 2026. `results/mi_adopters.csv` lists the adopting subdivisions, their earliest adoption dates and their sources as used in the manuscript.

## Running

Python 3.10 or later.

```
pip install -r requirements.txt
python3 paper_computations.py --out .
python3 michigan_illustration.py
```

`paper_computations.py` takes about one minute and uses a fixed random seed. `pygraphviz` requires Graphviz to be installed on the system.
