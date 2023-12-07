1) See hw1 if you'd like to see installation instructions. You do NOT have to redo them.
2) See the PDF for the rest of the instructions.



Commands:
Run Random Policy:
python cs285/scripts/run_hw5_explore.py \
    -cfg experiments/exploration/pointmass_easy_random.yaml \
    --dataset_dir datasets/

python cs285/scripts/run_hw5_explore.py \
    -cfg experiments/exploration/pointmass_medium_random.yaml \
    --dataset_dir datasets/

python cs285/scripts/run_hw5_explore.py \
    -cfg experiments/exploration/pointmass_hard_random.yaml \
    --dataset_dir datasets/


Run RND Policy:
python cs285/scripts/run_hw5_explore.py \
    -cfg experiments/exploration/pointmass_easy_rnd.yaml
    --dataset_dir datasets/

python cs285/scripts/run_hw5_explore.py \
    -cfg experiments/exploration/pointmass_medium_rnd.yaml \
    --dataset_dir datasets/

python cs285/scripts/run_hw5_explore.py \
    -cfg experiments/exploration/pointmass_hard_rnd.yaml \
    --dataset_dir datasets/
