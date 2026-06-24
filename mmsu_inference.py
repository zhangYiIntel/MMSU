import os
import sys
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add build directory to Python path to import pipeline module
BUILD_DIR = os.environ.get(
    "CP_BUILD_DIR",
    os.path.join(os.path.dirname(__file__), "..", "openvino.pipeline.mx", "build"),
)
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

from pipeline import OmniPipeline, OmniInput, OmniGenerationConfig, AudioSegment


def main():
    parser = argparse.ArgumentParser(description="MMSU Inference Script using OmniPipeline")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to model directory with pipeline.yaml")
    parser.add_argument('--output_jsonl', type=str, required=True, help="Path to save output JSONL file")
    parser.add_argument('--device', type=str, default="CPU", help="Device for inference (default: CPU)")
    parser.add_argument('--split', type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to process (for testing)")
    parser.add_argument('--start_index', type=int, default=0, help="Starting index for processing (default: 0)")
    parser.add_argument('--quiet', action='store_true', help="Suppress error messages for individual samples")
    args = parser.parse_args()

    output_file = args.output_jsonl
    split = args.split
    model_dir = args.model_dir
    device = args.device

    # ============================
    # Step 1: Load HuggingFace Dataset
    # ============================
    print(f"Loading MMSU dataset (split: {split})...")
    # Disable audio decoding to avoid torchcodec dependency - we'll handle it manually
    from datasets import Audio
    dataset = load_dataset("ddwang2000/MMSU", split=split)
    # Cast audio to raw format to prevent automatic decoding
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Select range of samples
    start_idx = args.start_index
    if args.max_samples:
        end_idx = min(start_idx + args.max_samples, len(dataset))
        dataset = dataset.select(range(start_idx, end_idx))
        print(f"Processing samples {start_idx} to {end_idx-1} ({len(dataset)} samples)")

    # ============================
    # Step 2: Load OmniPipeline
    # ============================
    print(f"Loading OmniPipeline from {model_dir} on device {device}...")
    try:
        pipe = OmniPipeline(model_dir, device)
    except Exception as e:
        print(f"Error loading OmniPipeline: {e}")
        sys.exit(1)

    # ============================
    # Step 3: Configure Generation
    # ============================
    gen_config = OmniGenerationConfig()
    # Configure generation parameters as needed
    # gen_config.max_new_tokens = 512
    # gen_config.temperature = 0.0

    # ============================
    # Step 4: Process Dataset
    # ============================
    import soundfile as sf
    import numpy as np

    print(f"Processing {len(dataset)} samples...")
    with open(output_file, "w") as fout:
        # Use tqdm with dynamic_ncols to reduce visual clutter
        iterator = tqdm(dataset, desc="Inference", unit="sample", dynamic_ncols=True,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for idx, item in enumerate(iterator):
            # Initialize variables for error handling
            audio_path = None
            question = item.get("question", "")
            choice_a = item.get("choice_a", "")
            choice_b = item.get("choice_b", "")
            choice_c = item.get("choice_c", "")
            choice_d = item.get("choice_d", "")
            task_name = item.get("task_name", "")
            output = ""

            try:
                # Extract audio data
                audio = item["audio"]
                audio_path = audio["path"]

                # Download and read audio file from HuggingFace if needed
                if audio_path.startswith("http"):
                    # Download the file
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        import urllib.request
                        urllib.request.urlretrieve(audio_path, tmp_file.name)
                        audio_array, sampling_rate = sf.read(tmp_file.name)
                        os.remove(tmp_file.name)
                else:
                    # Try to read from bytes if available
                    if "bytes" in audio and audio["bytes"] is not None:
                        import io
                        audio_array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]))
                    else:
                        # Fall back to reading from path
                        audio_array, sampling_rate = sf.read(audio_path)

                # Ensure float32 format and proper shape
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)

                # Handle multi-channel audio - convert to mono if needed
                if len(audio_array.shape) > 1:
                    # Convert stereo/multi-channel to mono by averaging
                    audio_array = np.mean(audio_array, axis=1).astype(np.float32)

                # Ensure 1D array
                audio_array = audio_array.flatten()

                task_name = item["task_name"]

                # ============================
                # Construct Prompt
                # ============================
                question = item["question"]

                question_prompts = (
                    "Choose the most suitable answer from options A, B, C, and D. "
                    "You must respond with only A, B, C, or D."
                )

                choice_a = item["choice_a"]
                choice_b = item["choice_b"]
                choice_c = item.get("choice_c", "")
                choice_d = item.get("choice_d", "")

                choices = (
                    f"A. {choice_a}\n"
                    f"B. {choice_b}\n"
                    f"C. {choice_c}\n"
                    f"D. {choice_d}"
                )

                instruction = f"{question_prompts}\n\nQuestion: {question}\n\n{choices}"

                # ============================
                # Prepare OmniInput
                # ============================
                omni_input = OmniInput()
                omni_input.text = instruction

                # Add audio segment
                audio_seg = AudioSegment()
                # Convert numpy array to Python list of floats
                audio_data_list = [float(x) for x in audio_array]
                audio_seg.data = audio_data_list
                audio_seg.sample_rate = int(sampling_rate)
                omni_input.audios = [audio_seg]

                # ============================
                # Run Model Inference
                # ============================
                result = pipe.generate(omni_input, gen_config)

                # Extract generated text
                if result.texts and len(result.texts) > 0:
                    output = result.texts[0]
                else:
                    output = ""

            except Exception as e:
                if not args.quiet:
                    print(f"\nError processing sample {idx} (id: {item.get('id', 'unknown')}): {e}")

            # ============================
            # Save result
            # ============================
            result_record = {
                "id": item.get("id", ""),
                "audio_path": audio_path if audio_path else "",
                "question": question,
                "choice_a": choice_a,
                "choice_b": choice_b,
                "choice_c": choice_c,
                "choice_d": choice_d,
                "answer_gt": item.get("answer_gt", ""),
                "response": output,
                "task_name": task_name,
                "category": item.get("category", ""),
                "sub-category": item.get("sub-category", ""),
                "sub-sub-category": item.get("sub-sub-category", ""),
                "linguistics_sub_discipline": item.get("linguistics_sub_discipline", ""),
            }

            fout.write(json.dumps(result_record, ensure_ascii=False) + "\n")

    print(f"\nInference complete. Results saved to {output_file}")


if __name__ == "__main__":
    # Example usage:
    # python mmsu_inference.py --model_dir /path/to/model --output_jsonl results.jsonl --device CPU
    main()
