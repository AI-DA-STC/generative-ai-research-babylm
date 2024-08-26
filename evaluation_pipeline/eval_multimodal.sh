#!/bin/bash

MODEL_PATH=$1
MODEL_BASENAME=$(basename $MODEL_PATH)

for task in {"winoground_filtered","vqa_filtered"}; do
	if [ $task = "winoground_filtered" ]; then
		image_src="facebook/winoground"
	elif [ $task = "vqa_filtered" ]; then
		image_src="HuggingFaceM4/VQAv2"
	fi

	python -m lm_eval --model hf \
		--model_args pretrained=$MODEL_PATH \
		--tasks $task \
		--device cuda:0 \
		--batch_size 64 \
		--output_path results/${task}/${MODEL_BASENAME}/${task}_results.json \
		--image_src $image_src
done