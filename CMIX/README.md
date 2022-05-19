# CMIX: Deep Multi-agent Reinforcement Learning with Peak and Average Constraints

This is a Pytorch implementation of CMIX on ECML-PKDD 2021. 

## Running CMIX

To run CMIX on the constrained blocker game task

```
./run_blockergame.sh
```

To run CMIX on the vehicular network routing optimization task

```
./run_vn.sh
```

To run the baselines of the vehicular network routing optimization task

```
python3 run_vn_baselines.py log/vn_baselines [your environment instance path, e.g., log/vn_CMIX-S/env.pickle]
```

If you have any questions, please post an issue or send an email to chenyiliu9@gmail.com.

## Citation

```
@inproceedings{liu2021cmix,
  title={CMIX: Deep Multi-agent Reinforcement Learning with Peak and Average Constraints},
  author={Liu, Chenyi and Nan Geng and Vaneet Aggarwal and Tian Lan and Yuan Yang and Mingwei Xu},
  booktitle={Proc. ECML-PKDD},
  year={2021}
}
```



