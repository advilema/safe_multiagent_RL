# example to run CMIX-M on blockergame task
python3 main.py --rl-model cql --mixer multi-qmix --policy-disc --log-dir blockergame_CMIX-M --batch-size 128 --application blocker --training-epochs 60000 --buffer-size 1000 --max-env-t 24 --epsilon-scheduler linear --tau 0.999 --loss-beta 1. --epsilon-finish 0.05
