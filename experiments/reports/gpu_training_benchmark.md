# GPU training benchmark (Phase 9E)

device: `cuda:0`

| arch | device | batch | samples/s |
| --- | --- | ---: | ---: |
| cnn | cpu | 1 | 65.6 |
| cnn | cuda:0 | 1 | 55.9 |
| cnn | cpu | 4 | 117.1 |
| cnn | cuda:0 | 4 | 170.1 |
| cnn | cpu | 16 | 118.9 |
| cnn | cuda:0 | 16 | 817.0 |
| cnn | cpu | 32 | 151.8 |
| cnn | cuda:0 | 32 | 1738.4 |
| graph | cpu | 1 | 58.2 |
| graph | cuda:0 | 1 | 24.1 |
| graph | cpu | 4 | 111.5 |
| graph | cuda:0 | 4 | 118.9 |
| graph | cpu | 16 | 112.4 |
| graph | cuda:0 | 16 | 479.9 |
| graph | cpu | 32 | 111.2 |
| graph | cuda:0 | 32 | 911.0 |
