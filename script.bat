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
@REM set "lrs= 0.0001 0.00001 0.000001"
@REM set "alphas= 3 1 0.5 0.3 0.1"
@REM set "betas= 5 3 1 0.5 0.3"
@REM set "threshes= 0.1 0.3 0.5 1 1.5 2 2.5 3"
@REM for %%a in (%alphas%) do (
@REM     for %%b in (%betas%) do (
@REM         echo Running: python main.py --num_data 200 --thresh1 --alpha %%a --beta %%b --lr 0.000001 --epochs 50000
@REM         python main.py --num_data 200 --thresh 1 --alpha %%a --beta %%b --lr 0.000001 --epochs 50000

@REM     )
@REM )
@REM set "alphas= 5 3 1 0.5 0.3 0.1"
@REM set "betas= 5 3 1 0.5 0.3 0.1"
@REM for %%a in (%alphas%) do (
@REM     for %%b in (%betas%) do (
@REM         echo Running: python main.py --num_data 200 --thresh 1 --alpha %%a --beta %%b --lr 0.000001 --epochs 50000
@REM         python main.py --num_data 200 --thresh 1 --alpha %%a --beta %%b --lr 0.000001 --epochs 50000

@REM     )
@REM )


set "threshes= 1 2 3 4 5 6 7 8 9 10"
for /L %%i in (11,1,20) do (
    for %%t in (%threshes%) do (
        echo Running: python main.py --num_data %%i --thresh %%t --alpha 1 --beta 5 --lr 0.000001 --epochs 50000
        python AdvGAN_main.py --num_data %%i --thresh %%t --alpha 1 --beta 5 --lr 0.000001 --epochs 50000
    )
)