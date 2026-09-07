# Data

Do not commit private CDT images to GitHub.

Recommended private dataset layout:

```text
D:/PrivateDatasets/cdt/
  no_dementia/
  light/
  medium/
  heavy/
```

The current baseline uses binary labels:

| Source folder | Binary label |
| --- | --- |
| `no_dementia` | `normal` |
| `light` | `impaired` |
| `medium` | `impaired` |
| `heavy` | `impaired` |

If your folders use different names, update `configs/config.yaml`.
