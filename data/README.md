# Dataset Instructions

This project uses the MURA dataset from Stanford ML Group:
👉 https://stanfordmlgroup.github.io/competitions/mura/

## Steps to prepare the dataset:

1. Go to the official [MURA download page](https://stanfordmlgroup.github.io/competitions/mura/).
2. Download the dataset (`MURA-v1.1.zip`) after agreeing to the license.
3. Place the downloaded file inside the `data/` folder of this project.

You should now have:
ECE228-MURA-CNN/
├── data/
│ └── MURA-v1.1.zip

4. **Unzip the file manually**, or run the provided script:

```bash
python data/unzip_mura.py
