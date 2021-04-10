import mne
from mne.datasets.sleep_physionet.age import fetch_data
import numpy as np
import os
from pathlib import Path


def preprocess_subject(data_path, annot_path, out_path):
    exclude = ['EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker', 'EEG Fpz-Cz']
    # EEG Fpz-Cz, EEG Pz-Oz
    raw = mne.io.read_raw_edf(data_path, exclude=exclude)
    annot = mne.read_annotations(annot_path)
    annot.crop(annot[1]['onset'] - 30 * 60, annot[-2]['onset'] + 30 * 60)
    raw.set_annotations(annot, emit_warning=False)

    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}
    event_id = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3/4': 4,
        'Sleep stage R': 5}

    events, _ = mne.events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)
    tmax = 30. - 1. / raw.info['sfreq']
    epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=0., tmax=tmax, baseline=None, on_missing='ignore')

    data = epochs.get_data()
    labels = epochs.events[:, 2] - 1
    base_name = os.path.basename(data_path)
    subj_idx, rec_idx = int(base_name[3:5]), int(base_name[5])
    new_path = os.path.join(out_path, f"{subj_idx}_{rec_idx}.npz")
    print("Processing: " + new_path)
    np.savez(new_path, data=data, labels=labels)


def preprocess_data(in_path, out_path="preprocessed", num_subjects=83):
    Path(out_path).mkdir(parents=True, exist_ok=True)

    mne.set_log_level('ERROR')
    subjects = range(num_subjects)
    files = fetch_data(subjects, path=in_path, on_missing='ignore')
    for data_path, annot_path in files:
        preprocess_subject(data_path, annot_path, out_path)


preprocess_data('C:\\Users\\Stas\\Downloads\\sleep-edf-database-expanded-1.0.0')
