CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_qa.py \
	--predict_file /home/v-lizimeng/data/WebSRC/websrc1.0_dev_.json \
	--root_dir /home/v-lizimeng/data/WebSRC \
	--model_name_or_path microsoft/markuplm-base \
	--output_dir /home/v-lizimeng/unilm/markuplm/examples/fine_tuning/run_websrc/results_qa \
	--do_eval \
	--eval_all_checkpoints \
	--max_query_length 42 \
	--max_seq_length 512 \
	--per_gpu_eval_batch_size 128
