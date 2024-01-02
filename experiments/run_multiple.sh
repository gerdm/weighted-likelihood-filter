n_runs=100
for p_error in 0 5 10 15 20 25 30 35 40 45 50
do
    time ipython regressions.py kin8nm $p_error $n_runs
done