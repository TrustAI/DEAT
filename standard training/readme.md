
An example of how to run our codes and reproduce our results in Tab.1.
```bash
cd 'standard training'
python mdeat_cifar10.py --data-dir <path of the CIFAR-10 dataset> --fname output_pre --save-model
```

`evaluation_cifar10.py` is provided to evaluate the saved checkpoint.

```
python evaluation_cifar10.py --data-dir <path of the CIFAR-10 dataset> --model-dir <output folder> --model-name <model's name>
```
`autoattack_cifar10.py` is provided to evaluate the saved checkpoint via the [AutoAttack v1](https://github.com/fra31/auto-attack).

```
python autoattack_cifar10.py --data-dir <path of the CIFAR-10 dataset> --model-dir <output folder> --model-name <model's name> --log-name <default is aa_score>
python aalog_reader.py --model-dir <output folder> --log-name <default is aa_score>
```

**Notes**:
- `mart.py` and `trades.py` are adopted from their original implementations, which is publicly released at [mart](https://github.com/YisenWang/MART/blob/master/mart.py) and [trades](https://github.com/yaodongyu/TRADES/blob/master/trades.py).

- The code for Friendly Adversarial training is publicly released at [here](https://github.com/zjfheart/Friendly-Adversarial-Training).

- Amata's code hasn't been released, So we re-implement it to make comparison. We also find that Amata's core strategy is equivalent to our DEAT-$d$, where $d$ is given by $|\frac{T}{K_\text{max}-K_\text{min}}|$, $T$ is the number of epochs, $K_\text{max}$ and $K_\text{min}$ are manually defined bounds on adversarial iterations. 

