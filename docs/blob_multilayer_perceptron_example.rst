Versioned Multilayer Perceptron Example
======================================

This page provides a full copy-paste runnable example (based on the test flow) for:

- saving dataset snapshots with ``save_or_update_dataset``,
- saving model checkpoints with ``save_or_update_ml_model``,
- saving/loading PyTorch checkpoints with ``save_or_update_torch_model`` and ``load_torch_model``,
- generating a workflow provenance report.

.. code-block:: python

   import pickle
   import random

   import torch
   import torch.nn as nn
   import torch.optim as optim

   from flowcept import Flowcept
   from flowcept.instrumentation.flowcept_task import flowcept_task, get_current_context_task_id


   def set_reproducibility(seed: int):
       """Simple deterministic setup. This is optional. Not required by Flowcept."""
       random.seed(seed)
       torch.manual_seed(seed)
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(seed)
       if hasattr(torch, "use_deterministic_algorithms"):
           torch.use_deterministic_algorithms(True, warn_only=True)
       if hasattr(torch.backends, "cudnn"):
           torch.backends.cudnn.deterministic = True
           torch.backends.cudnn.benchmark = False
       return {
           "seed": seed,
           "torch_deterministic_algorithms": True,
           "torch_cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False))
           if hasattr(torch.backends, "cudnn")
           else False,
           "torch_cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False))
           if hasattr(torch.backends, "cudnn")
           else False,
       }


   class MultiLayerPerceptron(nn.Module):
       def __init__(self, input_size=2, hidden_size=16):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(input_size, hidden_size),
               nn.ReLU(),
               nn.Linear(hidden_size, hidden_size),
               nn.ReLU(),
               nn.Linear(hidden_size, 1),
               nn.Sigmoid(),
           )

       def forward(self, x):
           return self.net(x)


   @flowcept_task(output_names=["x_train_shape", "y_train_shape", "x_val_shape", "y_val_shape"])
   def get_dataset(n_samples, split_ratio, reproducibility):
       """Generate synthetic binary-classification data and save dataset blob."""
       generator = torch.Generator().manual_seed(reproducibility["seed"])
       x = torch.cat(
           [
               torch.randn(n_samples // 2, 2, generator=generator) + 2,
               torch.randn(n_samples // 2, 2, generator=generator) - 2,
           ]
       )
       y = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)]).unsqueeze(1)
       n_train = int(n_samples * split_ratio)
       x_train, x_val = x[:n_train], x[n_train:]
       y_train, y_val = y[:n_train], y[n_train:]

       Flowcept.db.save_or_update_dataset(
           object={"x_train": x_train, "y_train": y_train, "x_val": x_val, "y_val": y_val},
           task_id=get_current_context_task_id(),
           custom_metadata={"n_samples": n_samples, "split_ratio": split_ratio, **reproducibility},
           save_data_in_collection=True,
           pickle=True,
           control_version=True,
       )
       return x_train, y_train, x_val, y_val


   def validate(model, criterion, x_val, y_val):
       model.eval()
       with torch.no_grad():
           outputs = model(x_val)
           loss = criterion(outputs, y_val)
           predictions = outputs.round()
           accuracy = (predictions.eq(y_val).sum().item()) / y_val.size(0)
       return loss.item(), accuracy


   @flowcept_task
   def train_and_validate(n_input_neurons, epochs, x_train, y_train, x_val, y_val, checkpoint_check=2):
       model = MultiLayerPerceptron(input_size=n_input_neurons, hidden_size=16)
       criterion = nn.BCELoss()
       optimizer = optim.SGD(model.parameters(), lr=0.1)

       best_val_loss = float("inf")
       ml_model_object_id = None
       torch_model_object_id = None
       current_task_id = get_current_context_task_id()

       for epoch in range(1, epochs + 1):
           model.train()
           outputs = model(x_train)
           loss = criterion(outputs, y_train)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           current_val_loss, _ = validate(model, criterion, x_val, y_val)
           if epoch % checkpoint_check == 0 and current_val_loss < best_val_loss:
               best_val_loss = current_val_loss
               checkpoint_meta = {"loss": best_val_loss, "checkpoint_epoch": epoch}

               # Generic path (format-agnostic)
               ml_model_object_id = Flowcept.db.save_or_update_ml_model(
                   object=model.state_dict(),
                   object_id=ml_model_object_id,
                   task_id=current_task_id,
                   custom_metadata=checkpoint_meta,
                   save_data_in_collection=True,
                   pickle=True,
                   control_version=True,
               )

               # PyTorch helper path
               torch_model_object_id = Flowcept.db.save_or_update_torch_model(
                   model=model,
                   object_id=torch_model_object_id,
                   task_id=current_task_id,
                   custom_metadata=checkpoint_meta,
                   control_version=True,
               )

       final_val_loss, final_val_accuracy = validate(model, criterion, x_val, y_val)
       return {
           "val_loss": final_val_loss,
           "val_accuracy": final_val_accuracy,
           "best_val_loss": best_val_loss,
           "ml_model_object_id": ml_model_object_id,
           "torch_model_object_id": torch_model_object_id,
       }


   def run_training(n_samples=120, split_ratio=0.8, n_input_neurons=2, epochs=4, seed=42):
       reproducibility = set_reproducibility(seed)
       x_train, y_train, x_val, y_val = get_dataset(n_samples, split_ratio, reproducibility)
       return train_and_validate(n_input_neurons, epochs, x_train, y_train, x_val, y_val)


   if __name__ == "__main__":
       reproducibility = set_reproducibility(42)

       with Flowcept(workflow_name="MLP Train", workflow_args=reproducibility) as flowcept:
           run_result = run_training(seed=reproducibility["seed"])
           workflow_id = flowcept.current_workflow_id

       # Load best generic ml_model checkpoint
       ml_model_blob = Flowcept.db.get_ml_model(run_result["ml_model_object_id"])
       ml_state_dict = pickle.loads(ml_model_blob.data)
       model_from_ml_model = MultiLayerPerceptron(input_size=2, hidden_size=16)
       model_from_ml_model.load_state_dict(ml_state_dict)
       model_from_ml_model.eval()

       # Load best torch helper checkpoint
       model_from_torch_helper = MultiLayerPerceptron(input_size=2, hidden_size=16)
       Flowcept.db.load_torch_model(model_from_torch_helper, run_result["torch_model_object_id"])
       model_from_torch_helper.eval()

       # Optional: generate markdown report for this workflow
       Flowcept.generate_report(
           output_path=f"./PROVENANCE_CARD_{workflow_id}.md",
           workflow_id=workflow_id,
       )


Notes
-----

- In production, choose one model-save path (generic or torch-specific).
- Version history is append-only in ``object_history`` when ``control_version=True``.
- Dataset snapshots are useful for reproducibility and workflow linkage.
