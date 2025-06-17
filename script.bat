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
@REM for /L %%i in (89,1,100) do (
@REM     python AdvGAN_main.py --num_data %%i --thresh 5 --alpha 0.3 --beta 5 --lr 0.000001 --epochs 10000
@REM     python AdvGAN_main.py --num_data %%i --thresh 5 --alpha 0.5 --beta 1 --lr 0.000001 --epochs 10000
@REM     @REM python AdvGAN_main.py --num_data %%i --thresh 5 --alpha 3 --beta 0.3 --lr 0.000001 --epochs 10000
@REM )
@REM set "lrs= 0.0001 0.00001 0.000001"
set "alphas= 5 3 1 0.5 0.3 0.1"
set "betas= 5 3 1 0.5 0.3 0.1"
set "threshes= 5"
for %%a in (%alphas%) do (
    for %%b in (%betas%) do (
        for %%t in (%threshes%) do (
            for /L %%i in (0,1,100) do (    
                echo Running: python AdvGAN_main.py --num_data %%i --thresh %%t --alpha %%a --beta %%b --lr 0.000001 --epochs 10000
                python AdvGAN_main.py --num_data %%i --thresh %%t --alpha %%a --beta %%b --lr 0.000001 --epochs 10000
            )
        )
    )
)
@REM set "alphas= 5 3 1 0.5 0.3 0.1"
@REM set "betas= 5 3 1 0.5 0.3 0.1"
@REM for %%a in (%alphas%) do (
@REM     for %%b in (%betas%) do (
@REM         echo Running: python main.py --num_data 200 --thresh 1 --alpha %%a --beta %%b --lr 0.000001 --epochs 50000
@REM         python main.py --num_data 200 --thresh 1 --alpha %%a --beta %%b --lr 0.000001 --epochs 50000

@REM     )
@REM )


@REM set "threshes= 1 2 3 4 5 6 7 8 9 10"
@REM for /L %%i in (1,1,1) do (
@REM     for %%t in (%threshes%) do (
@REM         echo Running: python main.py --num_data %%i --thresh %%t --alpha 1 --beta 5 --lr 0.000001 --epochs 50000
@REM         python AdvGAN_main.py --num_data %%i --thresh %%t --alpha 1 --beta 5 --lr 0.000001 --epochs 50000
@REM     )
@REM )