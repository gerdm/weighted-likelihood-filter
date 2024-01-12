n_runs=100
for p_error in 0 5 10 15 20 25 30 35 40 45
do
    time ipython regressions.py naval-propulsion-plant $p_error $n_runs
done