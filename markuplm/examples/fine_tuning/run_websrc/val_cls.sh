CUDA_VISIBLE_DEVICES=4,5,6,7 python run_cls.py \
	--train_file /home/v-lizimeng/data/WebSRC/websrc1.0_train_.json \
	--predict_file /home/v-lizimeng/data/WebSRC/websrc1.0_dev_.json \
	--root_dir /home/v-lizimeng/data/WebSRC \
	--model_name_or_path microsoft/markuplm-base \
	--output_dir /home/v-lizimeng/unilm/markuplm/examples/fine_tuning/run_websrc/results_cls \
	--do_eval \
	--eval_all_checkpoints \
	--max_query_length 42 \
	--max_seq_length 512 \
	--per_gpu_eval_batch_size 64 \
	--num_node_spans_per_case 384