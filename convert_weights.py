import os
import torch


def fix_efficientnet_keys(state_dict):
    """Maps torchvision EfficientNet keys to timm format perfectly."""
    timm_state_dict = {}
    for key, value in state_dict.items():
        new_key = key

        # Translate Stem & Head layers
        if key.startswith("features.0.0."):
            new_key = key.replace("features.0.0.", "conv_stem.")
        elif key.startswith("features.0.1."):
            new_key = key.replace("features.0.1.", "bn1.")
        elif key.startswith("features.7.0."):
            new_key = key.replace("features.7.0.", "conv_head.")
        elif key.startswith("features.7.1."):
            new_key = key.replace("features.7.1.", "bn2.")
        elif key.startswith("classifier.1."):
            new_key = key.replace("classifier.1.", "classifier.")

        if "num_batches_tracked" in key:
            continue

        # Deep block level matching (torchvision features.X.Y.block -> timm blocks.X.Y)
        if key.startswith("features."):
            parts = key.split(".")
            if len(parts) > 3 and parts[1].isdigit() and parts[2].isdigit():
                torchvision_stage = int(parts[1])
                block_idx = int(parts[2])

                timm_stage = torchvision_stage - 1

                remaining = parts[3:]
                if remaining[0] == "block":
                    remaining = remaining[1:]

                sub_path = ".".join(remaining)

                # Map specific convolution block layer types
                if timm_stage == 0:
                    sub_path = sub_path.replace("0.0.", "conv_dw.")
                    sub_path = sub_path.replace("0.1.", "bn1.")
                    sub_path = sub_path.replace("2.0.", "conv_pw.")
                    sub_path = sub_path.replace("2.1.", "bn2.")
                else:
                    sub_path = sub_path.replace("0.0.", "conv_pw.")
                    sub_path = sub_path.replace("0.1.", "bn1.")
                    sub_path = sub_path.replace("1.0.", "conv_dw.")
                    sub_path = sub_path.replace("1.1.", "bn2.")
                    sub_path = sub_path.replace("3.0.", "conv_pwl.")
                    sub_path = sub_path.replace("3.1.", "bn3.")

                # Translate Squeeze-and-Excitation layers
                if "fc1" in sub_path:
                    sub_path = sub_path.replace("fc1", "se.conv_reduce")
                if "fc2" in sub_path:
                    sub_path = sub_path.replace("fc2", "se.conv_expand")
                if "block.1.fc" in key or "block.2.fc" in key:
                    sub_path = sub_path.replace("1.se.", "se.")
                    sub_path = sub_path.replace("2.se.", "se.")

                new_key = f"blocks.{timm_stage}.{block_idx}.{sub_path}"

        timm_state_dict[new_key] = value
    return timm_state_dict


def convert_file(input_path, output_path):
    print(f"Reading: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")

    # Unpack state dict if nested inside a training checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Apply remapping conversion
    converted_dict = fix_efficientnet_keys(state_dict)

    # Save clean translated weights
    torch.save(converted_dict, output_path)
    print(f"🎉 Success! Converted weights saved to: {output_path}")


if __name__ == "__main__":
    # Change these filenames to match whatever you named your weights file!
    INPUT_FILE = "efficientnet-b3_best.pth"
    OUTPUT_FILE = "efficientnet-b3_best_timm.pth"

    convert_file(INPUT_FILE, OUTPUT_FILE)