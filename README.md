# Fine-tune VLM for Food and Drink Extraction

Fine-tune **SmolVLM2-500M-Video-Instruct** to identify food and drinks in images and generate structured JSON. This project walks through data preparation, supervised fine-tuning, loss visualization, and comparison of the base and fine-tuned models in a Gradio interface.

The complete workflow lives in [`main.ipynb`](main.ipynb).

## Task

Given an image, the model is prompted to return:

| Field | Type | Description |
| --- | --- | --- |
| `is_food` | Integer | `1` when food or drinks are visible; otherwise `0`. |
| `image_title` | String | A short food-related title, or an empty string when no food or drinks are present. |
| `food_items` | List of strings | Visible edible food items. |
| `drink_items` | List of strings | Visible drinks. |

Example target format (illustrative, not a measured model result):

```json
{
  "is_food": 1,
  "image_title": "Pizza and a glass of juice",
  "food_items": ["pizza"],
  "drink_items": ["juice"]
}
```

Outputs are generated text; the notebook does not validate them against a JSON schema.

## Project structure

```text
Fine-tune-VLM/
├── main.ipynb   # Data preparation, training, evaluation, and Gradio demo
└── README.md
```

## Setup

Use Jupyter locally or open the notebook in Google Colab with a GPU runtime. Internet access is needed to download packages, the model, and the dataset.

The notebook is configured for CUDA and BF16 training. Use a CUDA-enabled PyTorch installation and a GPU that supports BF16 for the configuration as written. The initial CPU fallback variable does not make the entire notebook CPU-compatible: an inference cell explicitly selects CUDA and training enables BF16.

In your notebook environment, install the dependencies:

```bash
python -m pip install torch transformers datasets accelerate trl pillow num2words matplotlib gradio numpy jupyterlab
```

The notebook's first cell installs only `trl` and `num2words`, so a fresh environment needs the additional packages above. Dependency versions are not pinned in this repository.

Start Jupyter and open `main.ipynb`:

```bash
jupyter lab main.ipynb
```

### Adjustments before running

- In the first base-model pipeline call, change `max_new_token=256` to `max_new_tokens=256`.
- On Python versions earlier than 3.12, the validation comparison cell needs different quote styles inside its f-string. Use:

  ```python
  print(f"[INFO] Example model ideal output:\n{random_val_sample_model_output['content'][0]['text']}")
  ```

- If your GPU does not support BF16, adapt both the model/pipeline dtypes and the trainer's precision settings to your hardware before training.

## Run the workflow

Execute the notebook cells in order:

1. **Load and inspect data.** Download `mrdbourke/FoodExtract-1k-Vision` using Hugging Face Datasets. Each example supplies an `image` and an `output_label_json` label.
2. **Build conversations.** Convert examples into system, user, and assistant messages, with an image and extraction prompt as input and serialized JSON as the target.
3. **Create splits.** Shuffle the dataset's training split with seed `42`, then divide it into 80% training and 20% validation.
4. **Inspect base-model predictions.** Run inference with a Transformers pipeline and with the processor's chat template.
5. **Prepare fine-tuning.** Freeze the vision encoder and use a custom collator to process images and text. Padding and `<image>` token labels are masked from the loss; other text tokens remain supervised.
6. **Train and save.** Run TRL's `SFTTrainer`, save the model, and plot training and validation losses.
7. **Compare outputs.** Generate predictions from both models on a validation image.
8. **Launch the demo.** Run the final Gradio cells and upload an image to view the two models' text outputs side by side.

## Training configuration

| Setting | Notebook value |
| --- | --- |
| Base model | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| Training examples used | First 50 examples of the shuffled training partition |
| Evaluation examples used | First 10 examples of the shuffled validation partition |
| Epochs | 2 |
| Per-device batch size | 1 for training and evaluation |
| Gradient accumulation | 4 steps |
| Learning rate | `2e-4` |
| Optimizer | `adamw_torch_fused` |
| Scheduler | Constant |
| Precision | BF16 |
| Gradient checkpointing | Enabled, with `use_reentrant=False` |
| Evaluation / checkpoint saving | Every epoch |
| Checkpoint retention setting | `save_total_limit=1` |
| Best-model loading | Enabled at the end of training |
| Hub upload / external reporting | Disabled |

The notebook updates parameters outside the frozen vision encoder; it does not configure LoRA or another adapter method.

To train on the full prepared partitions, replace `train_dataset[:50]` and `val_dataset[:10]` in the `SFTTrainer` cell with `train_dataset` and `val_dataset` respectively. Adjust training settings in the `SFTConfig` cell.

## Saved model and demo

Training checkpoints and the model saved by `trainer.save_model()` use this output directory:

```text
smolvlm-500m-FoodExtract-Vision-v1/
```

The comparison cells load the fine-tuned model from that directory. Run training and saving before those cells, or point `CHECKPOINT_DIR_NAME` to an existing compatible checkpoint. Keep the processor files with the checkpoint; if they are missing, save them with:

```python
processor.save_pretrained(training_args.output_dir)
```

The Gradio interface launches with `iface.launch(debug=True)` and displays **Pre-trained Model Output** and **Fine-tuned Model Output** for an uploaded image.

## Evaluation notes

The notebook provides loss curves and qualitative output comparisons. It does not compute extraction accuracy or JSON-validity metrics. The base comparison pipeline uses `do_sample=False`, while the fine-tuned pipeline uses `do_sample=True`; align these settings for a more controlled comparison.

Multiple model instances are created across the notebook. If GPU memory is exhausted, release unused pipelines/models or restart the runtime and execute only the needed sections.
