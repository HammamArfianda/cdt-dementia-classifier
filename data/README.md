# Data

Do not commit private CDT images to GitHub.

Recommended private dataset layout:

```text
D:/Projects/CDT Project/data/Dataset ISPO/
  Dataset Tidak Ada Demensia/
  Dataset Demensia Ringan/
  Dataset Demensia Sedang/
  Dataset Demensia Berat/
```

The current baseline uses binary labels:

| Source folder | Binary label |
| --- | --- |
| `Dataset Tidak Ada Demensia` | `normal` |
| `Dataset Demensia Ringan` | `impaired` |
| `Dataset Demensia Sedang` | `impaired` |
| `Dataset Demensia Berat` | `impaired` |

If your folders use different names, update `configs/config.yaml`.
