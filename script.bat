@echo off
@REM for /L %%i in (0,1,10) do (
@REM     python Modified_GAN_main.py --num_data %%i --lr 0.001
@REM     python Modified_GAN_main.py --num_data %%i --lr 0.0001
@REM     python Modified_GAN_main.py --num_data %%i --lr 0.00001
@REM )
@REM for /L %%i in (11,1,20) do (
@REM     python Modified_GAN_main.py --num_data %%i --alpha 10 --beta 1 --lr 0.00001 --thresh 0.5
@REM     python Modified_GAN_main.py --num_data %%i --alpha 5 --beta 5 --lr 0.00001 --thresh 0.5
@REM     python Modified_GAN_main.py --num_data %%i --alpha 1 --beta 10 --lr 0.00001 --thresh 0.5
@REM     python Modified_GAN_main.py --num_data %%i --alpha 0.5 --beta 0.5 --lr 0.00001 --thresh 0.5
@REM )
@REM for /L %%i in (21,1,30) do (
@REM     python Modified_GAN_main.py --num_data %%i --thresh 0.5
@REM     python Modified_GAN_main.py --num_data %%i --thresh 1
@REM     python Modified_GAN_main.py --num_data %%i --thresh 5
@REM     python Modified_GAN_main.py --num_data %%i --thresh 9
@REM )
@REM for /L %%i in (0,1,10) do (
@REM     python Modified_GAN_main.py --num_data %%i --thresh 2.5 --alpha 0.6 --beta 0.4 --lr 0.000001
@REM )
@REM echo Running: python Modified_GAN_main.py --num_data 12 --thresh 2.5 --alpha 0.6 --beta 0.4 --lr 0.000001
@REM python Modified_GAN_main.py --num_data 12 --thresh 2.5 --alpha 0.6 --beta 0.4 --lr 0.000001


@REM echo Running: python Modified_GAN_main.py --num_data 12 --thresh 2.5 --alpha 0.1 --beta 1 --lr 0.000001
@REM python Modified_GAN_main.py --num_data 12 --thresh 2.5 --alpha 0.1 --beta 1 --lr 0.000001


@REM echo Running: python Modified_GAN_main.py --num_data 12 --thresh 2.5 --alpha 0.01 --beta 1 --lr 0.000001
@REM python Modified_GAN_main.py --num_data 12 --thresh 2.5 --alpha 0.01 --beta 1 --lr 0.000001

@REM for /L %%i in (1,1,10) do (
@REM     echo Running: python Modified_GAN_main.py --num_data %%i --thresh 1.0 --alpha 1 --beta 0.1 --lr 0.000001 --epochs 10000
@REM     python Modified_GAN_main.py --num_data %%i --thresh 1.0 --alpha 1 --beta 0.1 --lr 0.000001 --epochs 10000
@REM     echo Running: python Modified_GAN_main.py --num_data %%i --thresh 1.0 --alpha 3 --beta 0.1 --lr 0.000001 --epochs 10000
@REM     python Modified_GAN_main.py --num_data %%i --thresh 1.0 --alpha 3 --beta 0.1 --lr 0.000001 --epochs 10000
@REM     echo Running: python Modified_GAN_main.py --num_data %%i --thresh 1.0 --alpha 1 --beta 0.1 --lr 0.000001 --epochs 10000
@REM     python Modified_GAN_main.py --num_data %%i --thresh 1.0 --alpha 1 --beta 0.1 --lr 0.000001 --epochs 10000
@REM     echo Running: python Modified_GAN_main.py --num_data %%i --thresh 1.0 --alpha 1 --beta 1 --lr 0.000001 --epochs 10000
@REM     python Modified_GAN_main.py --num_data %%i --thresh 1.0 --alpha 1 --beta 1 --lr 0.000001 --epochs 10000
@REM )

@REM set "alphas= 5 3 1 0.5 0.3 0.1"
@REM set "betas= 5 3 1 0.5 0.3 0.1"
@REM for %%a in (%alphas%) do (
@REM     for %%b in (%betas%) do (
@REM         echo Running: python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha %%a --beta %%b --lr 0.000001 --epochs 50000
@REM         python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha %%a --beta %%b --lr 0.000001 --epochs 50000

@REM     )
@REM )
python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha 0.1 --beta 0.5 --lr 0.000001 --epochs 50000
python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha 0.1 --beta 3 --lr 0.000001 --epochs 50000
python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha 0.5 --beta 0.1 --lr 0.000001 --epochs 50000
python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha 0.5 --beta 0.3 --lr 0.000001 --epochs 50000
python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha 0.5 --beta 1 --lr 0.000001 --epochs 50000
python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha 1 --beta 5 --lr 0.000001 --epochs 50000
python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha 3 --beta 3 --lr 0.000001 --epochs 50000
python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha 3 --beta 5 --lr 0.000001 --epochs 50000
@REM python Modified_GAN_main.py --num_data 200 --thresh 1 --alpha 1 --beta 1 --lr 0.000001 --epochs 50000