n_runs=100
p_error=10
for filename in ./data/*
do
    time ipython regressions.py "${filename##*/}" $p_error $n_runs
done