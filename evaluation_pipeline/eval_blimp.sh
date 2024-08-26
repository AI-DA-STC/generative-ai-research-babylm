#!/bin/bash

CONFIG_DIR="/Users/krishnaiyer/generative-ai-research-babylm/conf/eval_config_ensemble"
OUTPUT_BASE_DIR="results/blimp/GPT2"

process_yaml() {
    local config_path="$1"
    echo "Running evaluation with config: $config_path"
    
    # Extract the relative path from CONFIG_DIR
    local rel_path=${config_path#$CONFIG_DIR/}
    # Remove the .yaml extension
    local file_path_no_ext=${rel_path%.yaml}
    # Replace directory separators with underscores
    local output_name=${file_path_no_ext//\//_}
    
    # Create output path using the modified config filename
    local output_path="$OUTPUT_BASE_DIR/${output_name}_ensemble_results.json"
    
    python -m lm_eval --model gpt2-custom \
    --model_args config_path="$config_path" \
    --tasks blimp_filtered,blimp_supplement,glue \
    --output_path "$output_path"
    
    echo "Evaluation complete for $config_path"
    echo "Results saved to $output_path"
    echo "----------------------------------------"
}
while IFS= read -r -d '' config_file
do
    process_yaml "$config_file"
done < <(find "$CONFIG_DIR" -type f -name "*yaml" -print0)

echo "All evaluations completed."

# use `--model_args pretrained=$MODEL_PATH,backend="mlm"` if you're using a custom masked LM
# add --trust_remote_code if you need to load custom config/model files
