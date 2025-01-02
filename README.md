# Charisardo

## Setup

### Submodules

```bash
git submodule update --init --recursive
```

### Install pokemon-vgc-engine

From [https://gitlab.com/DracoStriker/pokemon-vgc-engine](https://gitlab.com/DracoStriker/pokemon-vgc-engine)

Run the following commands to install vgc as a package in your venv:

```bash
cd pokemon-vgc-engine
pip install .
```

### Project files

#### Report

[**report_notebook.ipynb**](report_notebook.ipynb) contains the report of the project.

#### Data generation

- [**report_data.py**](test/report_data.py) simulate all the combination of battles and save the data in json files.
- [**graphics.ipynb**](test/graphics.ipynb) use the data generated to create graphics.

#### Policies

- [**Heuristical.py**](policies/Heuristical.py) Heuristical policy.
- [**PrunedTreeSearch.py**](policies/PrunedTreeSearch.py) Pruned Tree Search policy.
- [**WeightedGreedy.py**](policies/WeightedGreedy.py) Weighted Greedy policy.

#### Utils

- [**own_env.py**](utils/own_env.py) custom changes to the pokemon environment.
- [**own_team_generator.py**](utils/own_team_generator.py) custom team generator.
- [**scraping_data.py**](utils/scraping_data.py) utilies function to simulate the battle also using parallel pools.

### How to test the UX interface

```bash
cd pokemon-vgc-engine/vgc/ux
# Start the UX interface
python PkmBattleUX.py
```

In another terminal:

```bash
cd pokemon-vgc-engine/vgc/ux
# Start the GUI
python PkmBattleClientTest.py
```
