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

### Test the engine

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