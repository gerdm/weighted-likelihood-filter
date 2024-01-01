for p_error in 45 50 60 65
do
    time ipython regressions.py concrete $p_error
done