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
@REM for /L %%i in (0,1,10) do (
@REM     python main.py --num_data %%i --thresh 2.5 --alpha 0.6 --beta 0.4 --lr 0.000001
@REM )
@REM echo Running: python main.py --num_data 12 --thresh 2.5 --alpha 0.6 --beta 0.4 --lr 0.000001
@REM python main.py --num_data 12 --thresh 2.5 --alpha 0.6 --beta 0.4 --lr 0.000001


@REM echo Running: python main.py --num_data 12 --thresh 2.5 --alpha 0.1 --beta 1 --lr 0.000001
@REM python main.py --num_data 12 --thresh 2.5 --alpha 0.1 --beta 1 --lr 0.000001


@REM echo Running: python main.py --num_data 12 --thresh 2.5 --alpha 0.01 --beta 1 --lr 0.000001
@REM python main.py --num_data 12 --thresh 2.5 --alpha 0.01 --beta 1 --lr 0.000001


echo Running: python main.py --num_data 1 --thresh 1.0 --alpha 0.1 --beta 1 --lr 0.000001 --epochs 10000
python main.py --num_data 14 --thresh 1.0 --alpha 2 --beta 1 --lr 0.000001 --epochs 10000
echo Running: python main.py --num_data 2 --thresh 1.0 --alpha 1 --beta 1 --lr 0.000001 --epochs 10000
python main.py --num_data 15 --thresh 1.0 --alpha 3 --beta 1 --lr 0.000001 --epochs 10000
echo Running: python main.py --num_data 3 --thresh 1.0 --alpha 1 --beta 0.1 --lr 0.000001 --epochs 10000
python main.py --num_data 16 --thresh 1.0 --alpha 5 --beta 1 --lr 0.000001 --epochs 10000