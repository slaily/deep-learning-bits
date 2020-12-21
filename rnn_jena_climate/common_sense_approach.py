"""
Computing the common-sense baseline MAE
"""
def evaluate_naive_method():
    batch_maes = []

    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)

    print(f'Batch MAEs: {batch_maes}')


evaluate_naive_method()