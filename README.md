## 🤖 About

## 🚀 Tutorial

Before starting, the following packages must be installed:

```bash
pip install retnext
pip install pymoxel>=0.4.0
pip install aidsorb>=2.0.0
```

> [!NOTE]
> **All examples below assume the use of the pretrained model**. Therefore, the image generation and preprocessing parameters are configured accordingly.
> If you want to train the model from scratch, you can follow a similar procedure.

### 🎨 Generate the energy images

You can generate the energy images from the CLI as following:

```bash
moxel path/to/CIFs path/to/voxels_data/ --grid_size=32 --cubic_box=30
```
Alternatively, for more fine-grained control over the materials to be processed you can use ``voxels_from_files``:
```python
from moxel.utils import voxels_from_files

cifs = ['foo.cif', 'bar.cif', ...]
voxels_from_files(cifs, 'path/to/voxels_data/', grid_size=32, cubic_box=30)
```

### ❄️ Use RetNeXt as feature extractor

Each energy image is passed through pretrained model to extract
```python
import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate_fn_map
from torchvision.transforms.v2 import Compose
from retnext.modules import RetNeXt
from retnext.transforms import AddChannelDim, BoltzmannFactor
from aidsorb.data import PCDDataset as VoxelsDataset


# Required for collating unlabeled samples
def collate_none(batch, *, collate_fn_map):
    return None


# Get the names of the materials
names = [f.removesuffix('.npy') for f in os.listdir('path/to/voxels_data/')]

# Preprocessing transformations
transform_x = Compose([AddChannelDim(), BoltzmannFactor()])

# Create the dataset
dataset = VoxelsDataset(names, path_to_X='path_to_voxels_data, transform_x=transform_x)

# Create the dataloader (adjust batch_size and num_workers)
dataloader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=256, num_workers=8)
default_collate_fn_map.update({NoneType: collate_none})

# Load pretrained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RetNeXt(pretrained=True).to(device)

# Freeze the model
model.requires_grad_(False)
model.eval()
model.fc = torch.nn.Identity()  # So .forward() returns the embeddings.

# Extract features
Z = torch.cat([model(x.to(device)) for x, _ in dataloader])

# Store features as .csv file
df = pd.DataFrame(Z.numpy(), index=names)
df.to_csv(f'emdeddings.csv', index=True, index_label='name')
```

### 🔥 Fine-tune RetNeXt

```python
from torchvision.transforms.v2 import Compose, RandomChoice

# Freeze the model
model.requires_grad_(False)
model.eval()
model.fc = torch.nn.Identity() # So .forward() returns the embeddings.
```

### 📑 Citing

### ⚖️ License
