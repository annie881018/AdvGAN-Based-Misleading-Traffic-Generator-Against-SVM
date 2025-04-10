@echo off
@REM for /L %%i in (0,1,10) do (
@REM     python main.py --num_data %%i --lr 0.001
@REM     python main.py --num_data %%i --lr 0.0001
@REM     python main.py --num_data %%i --lr 0.00001
@REM )
@REM for /L %%i in (11,1,20) do (
@REM     python main.py --num_data %%i --alpha 10 --beta 1 --lr 0.00001 --thresh 0.5
@REM     python main.py --num_data %%i --alpha 5 --beta 5 --lr 0.00001 --thresh 0.5
@REM     python main.py --num_data %%i --alpha 1 --beta 10 --lr 0.00001 --thresh 0.5
@REM     python main.py --num_data %%i --alpha 0.5 --beta 0.5 --lr 0.00001 --thresh 0.5
@REM )
@REM for /L %%i in (21,1,30) do (
@REM     python main.py --num_data %%i --thresh 0.5
@REM     python main.py --num_data %%i --thresh 1
@REM     python main.py --num_data %%i --thresh 5
@REM     python main.py --num_data %%i --thresh 9
@REM )
set "lrs= 0.0001 0.00001 0.000001"
set "alphas= 10 5 3 1 0.5 0.3 0.1"
set "betas= 10 5 3 1 0.5 0.3 0.1"
set "threshes= 0.1 0.3 0.5 1 1.5 2 2.5 3"
for %%l in (%lrs%) do (
    for %%a in (%alphas%) do (
        for %%b in (%betas%) do (
            for %%t in (%threshes%) do (
                echo Running: python main.py --num_data 200 --thresh %%t --alpha %%a --beta %%b --lr %%l --epochs 5000
                python main.py --num_data 200 --thresh %%t --alpha %%a --beta %%b --lr %%l --epochs 5000
            )
        )
    )
)