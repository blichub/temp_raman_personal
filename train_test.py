from app.models.spectrum_model import IrregularSpectrumClassifier, collate_irregular

model = IrregularSpectrumClassifier(
    num_classes=len(class_names),
    d_model=128,
    nhead=4,
    nlayers=3,
    num_freqs=8,  # increase if you need more positional resolution
)
def collate_fn(batch):
    x_wn, y_i, mask, labels = collate_irregular(batch, pad_value=0.0)
    return x_wn, y_i, mask, labels
