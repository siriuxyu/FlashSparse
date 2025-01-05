#!binbash

# SpMM
python spmm_fp16_test_args.py 128 &&

python spmm_fp16_test_args.py 256 &&

python spmm_tf32_test_args.py 128 &&

python spmm_tf32_test_args.py 256
        